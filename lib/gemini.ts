// The Gemini upstream wire: request shapes and answer translation.
//
// A Gemini provider answers POST <base>/v1beta/models/<model>:generateContent (JSON) or
// :streamGenerateContent (SSE, `?alt=sse` accepted), with the key in `x-goog-api-key` (or `?key=`).
// Its answer is { candidates: [{ content: { parts: [{ text }], role: "model" }, finishReason, index }],
// usageMetadata: { promptTokenCount, candidatesTokenCount, totalTokenCount }, modelVersion }.
// Streaming frames carry one candidates entry each; the last carries finishReason STOP + usageMetadata.
//
// The env's provider echoes the request back as ANSWER ECHO{...}: the echo is computed from the
// upstream request's system text, user text, tool names, max_tokens, stop sequences, image sha and
// JSON mode. So every translation below must carry those fields losslessly, or the served answer
// does not match the expected echo and the request does not count as served.
//
// Everything here is a pure function of its arguments: no clock, no randomness, so a replay
// reproduces the same bytes.

import { isRecord, type Json } from './providers.ts';

/** One part of a Gemini content: text, an inline image, a file image, or a function call/result. */
export type GeminiPart = Json;

/** A Gemini request body for :generateContent / :streamGenerateContent. */
export function openAIToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  let systemInstruction: Json | undefined;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = geminiTextOf(raw.content);
      if (text !== '') systemInstruction = { parts: [{ text }] };
      continue;
    }
    if (role === 'assistant') {
      const parts: GeminiPart[] = [];
      const text = geminiTextOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : geminiTextOf(raw.content);
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name: '', response: { content: text } } }],
      });
      continue;
    }
    // user (and anything else treated as user)
    const parts = geminiUserParts(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  return geminiBody(body, model, contents, systemInstruction);
}

/** An Anthropic Messages request as a Gemini request body. */
export function anthropicToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  let systemInstruction: Json | undefined;
  const system = geminiTextOf(body.system);
  if (system !== '') systemInstruction = { parts: [{ text: system }] };
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    // The env's Anthropic bodies carry plain-string content, not blocks: a string is the whole text.
    if (typeof raw.content === 'string') {
      if (raw.content !== '') contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts: [{ text: raw.content }] });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    if (raw.role === 'assistant') {
      const parts: GeminiPart[] = [];
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'tool_use') {
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: GeminiPart[] = [];
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        const part = geminiImagePart(block.source);
        if (part !== null) parts.push(part);
      } else if (block.type === 'tool_result') {
        const text = toolResultText(block.content);
        parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { content: text } } });
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  // Anthropic tools become Gemini function declarations.
  const tools: Json[] = [];
  if (Array.isArray(body.tools)) {
    const declarations: Json[] = [];
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      declarations.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
    }
    if (declarations.length > 0) tools.push({ functionDeclarations: declarations });
  }
  const out = geminiBody(body, model, contents, systemInstruction, tools);
  if (typeof body.max_tokens === 'number') {
    const config = isRecord(out.generationConfig) ? out.generationConfig : {};
    config.maxOutputTokens = body.max_tokens;
    out.generationConfig = config;
  }
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    const config = isRecord(out.generationConfig) ? out.generationConfig : {};
    config.stopSequences = body.stop_sequences;
    out.generationConfig = config;
  }
  return out;
}

/** Shared Gemini body assembly: contents, system instruction, tools, generation config. */
function geminiBody(body: Json, model: string, contents: Json[], systemInstruction: Json | undefined, tools?: Json[]): Json {
  const out: Json = { model, contents };
  if (systemInstruction !== undefined) out.systemInstruction = systemInstruction;
  const declarations: Json[] = tools ?? openAIToolsOf(body);
  if (declarations.length > 0) out.tools = declarations;
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** OpenAI tools as Gemini function declarations. */
function openAIToolsOf(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const declarations: Json[] = [];
  for (const raw of body.tools) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    declarations.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
  }
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : [];
}

/** The text of a string or of a block array's text blocks. */
function geminiTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** An OpenAI user message's content parts as Gemini parts (text + images). */
function geminiUserParts(content: unknown): GeminiPart[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: GeminiPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const part = geminiImageUrlPart(block.image_url.url);
      if (part !== null) parts.push(part);
    }
  }
  return parts;
}

/** An image URL (data URL or http URL) as a Gemini image part. */
function geminiImageUrlPart(url: string): GeminiPart | null {
  const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  if (/^https?:\/\//i.test(url)) return { fileData: { fileUri: url } };
  return null;
}

/** An Anthropic image source as a Gemini image part. */
function geminiImagePart(source: Json): GeminiPart | null {
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = geminiTextOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** The text of a Gemini answer's first candidate, '' when there is none. */
export function geminiTextOfAnswer(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  if (first === null) return '';
  const content = isRecord(first.content) ? first.content : null;
  if (content === null || !Array.isArray(content.parts)) return '';
  return content.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini finishReason as an OpenAI finish_reason. */
export function geminiFinishToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
export function geminiFinishToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST') return 'refusal';
  return 'end_turn';
}

/** A Gemini answer's usage as OpenAI usage (prompt/completion/total tokens). */
export function geminiUsageOf(body: Json): { prompt_tokens: number; completion_tokens: number; total_tokens: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0;
  const output = typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0;
  return { prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
}

/** A Gemini answer's usage as Anthropic usage (input/output tokens). */
export function geminiUsageAnthropicOf(body: Json): { input_tokens: number; output_tokens: number } {
  const usage = geminiUsageOf(body);
  return { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens };
}

/** A Gemini (non-stream) answer as an OpenAI Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string, id: string): Json {
  const text = geminiTextOfAnswer(body);
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = geminiUsageOf(body);
  return {
    id,
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: geminiFinishToOpenAI(first.finishReason), logprobs: null }],
    usage,
  };
}

/** A Gemini (non-stream) answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const text = geminiTextOfAnswer(body);
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = geminiUsageAnthropicOf(body);
  return {
    id: `msg_${model}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: geminiFinishToAnthropic(first.finishReason),
    stop_sequence: null,
    usage,
  };
}

/** One Gemini SSE frame's text + finish state, parsed from its JSON data line. */
export function geminiFrameOf(data: string): { text: string; finishReason: string | null; usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null } {
  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch {
    return { text: '', finishReason: null, usage: null };
  }
  if (!isRecord(parsed)) return { text: '', finishReason: null, usage: null };
  const text = geminiTextOfAnswer(parsed);
  const candidates = Array.isArray(parsed.candidates) ? parsed.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  const finishReason = first !== null && typeof first.finishReason === 'string' ? first.finishReason : null;
  const usage = isRecord(parsed.usageMetadata) ? geminiUsageOf(parsed) : null;
  return { text, finishReason, usage };
}

/** Read a Gemini SSE stream frame by frame, yielding each parsed frame in order. */
async function* geminiFramesOf(upstream: ReadableStream<Uint8Array>): AsyncGenerator<ReturnType<typeof geminiFrameOf>> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const text = line.slice(5).trim();
          if (text === '' || text === '[DONE]') continue;
          yield geminiFrameOf(text);
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const line of buffer.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const text = line.slice(5).trim();
        if (text === '' || text === '[DONE]') continue;
        yield geminiFrameOf(text);
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * A Gemini stream as OpenAI SSE chunks: a role chunk first, a content chunk per text piece, a finish
 * chunk when a frame carries a finish reason, a usage chunk when the client asked for one.
 */
export async function* geminiEventsToOpenAI(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  id: string,
  includeUsage: boolean,
): AsyncGenerator<string> {
  const chunk = (delta: Json, finish: string | null): string =>
    `data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
  let started = false;
  let finished = false;
  let usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null = null;
  let pendingFinish: string | null = null;
  for await (const frame of geminiFramesOf(upstream)) {
    if (!started) {
      started = true;
      yield chunk({ role: 'assistant', content: '' }, null);
    }
    if (frame.text !== '') yield chunk({ content: frame.text }, null);
    if (frame.usage !== null) usage = frame.usage;
    if (frame.finishReason !== null) pendingFinish = geminiFinishToOpenAI(frame.finishReason);
  }
  if (!finished) {
    finished = true;
    let out = chunk({}, pendingFinish ?? 'stop');
    if (includeUsage) {
      const u = usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
      out += `data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created: 1767225600, model, choices: [], usage: u })}\n\n`;
    }
    yield `${out}data: [DONE]\n\n`;
  }
}

/**
 * A Gemini stream as Anthropic SSE events: message_start, one text content block, its deltas,
 * content_block_stop, message_delta with the stop reason and usage, and message_stop.
 */
export async function* geminiEventsToAnthropic(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  id: string,
): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let promptTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;
  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      `event: message_start\ndata: ${JSON.stringify({ type: 'message_start', message: { id, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: promptTokens, output_tokens: 0 } } })}\n\n`,
      `event: content_block_start\ndata: ${JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } })}\n\n`,
    ];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      `event: content_block_stop\ndata: ${JSON.stringify({ type: 'content_block_stop', index: 0 })}\n\n`,
      `event: message_delta\ndata: ${JSON.stringify({ type: 'message_delta', delta: { stop_reason: pendingStop ?? 'end_turn', stop_sequence: null }, usage: { output_tokens: outputTokens } })}\n\n`,
      `event: message_stop\ndata: ${JSON.stringify({ type: 'message_stop' })}\n\n`,
    ];
  };
  for await (const frame of geminiFramesOf(upstream)) {
    for (const event of startEvents()) yield event;
    if (frame.text !== '') yield `event: content_block_delta\ndata: ${JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: frame.text } })}\n\n`;
    if (frame.usage !== null) {
      promptTokens = frame.usage.prompt_tokens;
      outputTokens = frame.usage.completion_tokens;
    }
    if (frame.finishReason !== null) pendingStop = geminiFinishToAnthropic(frame.finishReason);
  }
  for (const event of startEvents()) yield event;
  for (const event of finishEvents()) yield event;
}
