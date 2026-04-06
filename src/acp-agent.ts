import {
  Agent,
  AgentSideConnection,
  AuthenticateRequest,
  AuthMethod,
  AvailableCommand,
  CancelNotification,
  ClientCapabilities,
  ForkSessionRequest,
  ForkSessionResponse,
  InitializeRequest,
  InitializeResponse,
  ListSessionsRequest,
  ListSessionsResponse,
  LoadSessionRequest,
  LoadSessionResponse,
  ndJsonStream,
  NewSessionRequest,
  NewSessionResponse,
  PromptRequest,
  PromptResponse,
  ReadTextFileRequest,
  ReadTextFileResponse,
  RequestError,
  ResumeSessionRequest,
  ResumeSessionResponse,
  SessionConfigOption,
  SessionModelState,
  SessionModeState,
  SessionNotification,
  SetSessionConfigOptionRequest,
  SetSessionConfigOptionResponse,
  SetSessionModelRequest,
  SetSessionModelResponse,
  SetSessionModeRequest,
  SetSessionModeResponse,
  CloseSessionRequest,
  CloseSessionResponse,
  TerminalHandle,
  TerminalOutputResponse,
  WriteTextFileRequest,
  WriteTextFileResponse,
  StopReason,
} from "@agentclientprotocol/sdk";
import {
  CanUseTool,
  getSessionMessages,
  listSessions,
  McpServerConfig,
  ModelInfo,
  ModelUsage,
  Options,
  PermissionMode,
  Query,
  query,
  SDKPartialAssistantMessage,
  SDKResultMessage,
  SDKUserMessage,
  SlashCommand,
} from "@anthropic-ai/claude-agent-sdk";
import { ContentBlockParam } from "@anthropic-ai/sdk/resources";
import { BetaContentBlock, BetaRawContentBlockDelta } from "@anthropic-ai/sdk/resources/beta.mjs";
import { randomUUID } from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import packageJson from "../package.json" with { type: "json" };
import { SettingsManager } from "./settings.js";
import {
  ClaudePlanEntry,
  createPostToolUseHook,
  planEntries,
  registerHookCallback,
  toolInfoFromToolUse,
  toolUpdateFromEditToolResponse,
  toolUpdateFromToolResult,
} from "./tools.js";
import { nodeToWebReadable, nodeToWebWritable, Pushable, unreachable } from "./utils.js";

export const CLAUDE_CONFIG_DIR =
  process.env.CLAUDE_CONFIG_DIR ?? path.join(os.homedir(), ".claude");

const MAX_TITLE_LENGTH = 256;

function sanitizeTitle(text: string): string {
  // Replace newlines and collapse whitespace
  const sanitized = text
    .replace(/[\r\n]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (sanitized.length <= MAX_TITLE_LENGTH) {
    return sanitized;
  }
  return sanitized.slice(0, MAX_TITLE_LENGTH - 1) + "…";
}

/**
 * Logger interface for customizing logging output
 */
export interface Logger {
  log: (...args: any[]) => void;
  error: (...args: any[]) => void;
}

type AccumulatedUsage = {
  inputTokens: number;
  outputTokens: number;
  cachedReadTokens: number;
  cachedWriteTokens: number;
};

type Session = {
  query: Query;
  input: Pushable<SDKUserMessage>;
  cancelled: boolean;
  cwd: string;
  settingsManager: SettingsManager;
  accumulatedUsage: AccumulatedUsage;
  modes: SessionModeState;
  models: SessionModelState;
  configOptions: SessionConfigOption[];
  promptRunning: boolean;
  pendingMessages: Map<string, { resolve: (cancelled: boolean) => void; order: number }>;
  nextPendingOrder: number;
  abortController: AbortController;
};

type BackgroundTerminal =
  | {
      handle: TerminalHandle;
      status: "started";
      lastOutput: TerminalOutputResponse | null;
    }
  | {
      status: "aborted" | "exited" | "killed" | "timedOut";
      pendingOutput: TerminalOutputResponse;
    };

/**
 * Extra metadata that can be given when creating a new session.
 */
export type NewSessionMeta = {
  claudeCode?: {
    /**
     * Options forwarded to Claude Code when starting a new session.
     * Those parameters will be ignored and managed by ACP:
     *   - cwd
     *   - includePartialMessages
     *   - allowDangerouslySkipPermissions
     *   - permissionMode
     *   - canUseTool
     *   - executable
     * Those parameters will be used and updated to work with ACP:
     *   - hooks (merged with ACP's hooks)
     *   - mcpServers (merged with ACP's mcpServers)
     *   - disallowedTools (merged with ACP's disallowedTools)
     *   - tools (passed through; defaults to claude_code preset if not provided)
     */
    options?: Options;
  };
  additionalRoots?: string[];
};

/**
 * Extra metadata for 'gateway' authentication requests.
 */
type GatewayAuthMeta = {
  /**
   * These parameters are mapped to environment variables to:
   * - Redirect API calls via baseUrl
   * - Inject custom headers
   * - Bypass the default Claude login requirement
   */
  gateway: {
    baseUrl: string;
    headers: Record<string, string>;
  };
};

/**
 * Extra metadata that the agent provides for each tool_call / tool_update update.
 */
export type ToolUpdateMeta = {
  claudeCode?: {
    /* The name of the tool that was used in Claude Code. */
    toolName: string;
    /* The structured output provided by Claude Code. */
    toolResponse?: unknown;
  };
  /* Terminal metadata for Bash tool execution, matching codex-acp's _meta protocol. */
  terminal_info?: {
    terminal_id: string;
  };
  terminal_output?: {
    terminal_id: string;
    data: string;
  };
  terminal_exit?: {
    terminal_id: string;
    exit_code: number;
    signal: string | null;
  };
};

export type ToolUseCache = {
  [key: string]: {
    type: "tool_use" | "server_tool_use" | "mcp_tool_use";
    id: string;
    name: string;
    input: unknown;
  };
};

function isStaticBinary(): boolean {
  return process.env.CLAUDE_AGENT_ACP_IS_SINGLE_FILE_BUN !== undefined;
}

export async function claudeCliPath(): Promise<string> {
  return isStaticBinary()
    ? (await import("@anthropic-ai/claude-agent-sdk/embed")).default
    : import.meta.resolve("@anthropic-ai/claude-agent-sdk").replace("sdk.mjs", "cli.js");
}

function shouldHideClaudeAuth(): boolean {
  return process.argv.includes("--hide-claude-auth");
}

// Bypass Permissions doesn't work if we are a root/sudo user
const IS_ROOT = (process.geteuid?.() ?? process.getuid?.()) === 0;
const ALLOW_BYPASS = !IS_ROOT || !!process.env.IS_SANDBOX;

// Slash commands that the SDK handles locally without replaying the user
// message and without invoking the model.
const LOCAL_ONLY_COMMANDS = new Set([
  "/context", "/heapdump", "/extra-usage",
  "/compact", "/help", "/config", "/cost",
  "/login", "/logout", "/status", "/memory",
]);

const PERMISSION_MODE_ALIASES: Record<string, PermissionMode> = {
  auto: "auto",
  default: "default",
  acceptedits: "acceptEdits",
  dontask: "dontAsk",
  plan: "plan",
  bypasspermissions: "bypassPermissions",
  bypass: "bypassPermissions",
};

export function resolvePermissionMode(defaultMode?: unknown): PermissionMode {
  if (defaultMode === undefined) {
    return "default";
  }

  if (typeof defaultMode !== "string") {
    throw new Error("Invalid permissions.defaultMode: expected a string.");
  }

  const normalized = defaultMode.trim().toLowerCase();
  if (normalized === "") {
    throw new Error("Invalid permissions.defaultMode: expected a non-empty string.");
  }

  const mapped = PERMISSION_MODE_ALIASES[normalized];
  if (!mapped) {
    throw new Error(`Invalid permissions.defaultMode: ${defaultMode}.`);
  }

  if (mapped === "bypassPermissions" && !ALLOW_BYPASS) {
    throw new Error(
      "Invalid permissions.defaultMode: bypassPermissions is not available when running as root.",
    );
  }

  return mapped;
}

// Implement the ACP Agent interface
export class ClaudeAcpAgent implements Agent {
  sessions: {
    [key: string]: Session;
  };
  client: AgentSideConnection;
  toolUseCache: ToolUseCache;
  backgroundTerminals: { [key: string]: BackgroundTerminal } = {};
  clientCapabilities?: ClientCapabilities;
  logger: Logger;
  gatewayAuthMeta?: GatewayAuthMeta;

  constructor(client: AgentSideConnection, logger?: Logger) {
    this.sessions = {};
    this.client = client;
    this.toolUseCache = {};
    this.logger = logger ?? console;
  }

  async initialize(request: InitializeRequest): Promise<InitializeResponse> {
    this.clientCapabilities = request.clientCapabilities;

    // Bypasses standard auth by routing requests through a custom Anthropic-protocol gateway.
    // Only offered when the client advertises `auth._meta.gateway` capability.
    const supportsGatewayAuth = request.clientCapabilities?.auth?._meta?.gateway === true;

    const gatewayAuthMethod: AuthMethod = {
      id: "gateway",
      name: "Custom model gateway",
      description: "Use a custom gateway to authenticate and access models",
      _meta: {
        gateway: {
          protocol: "anthropic",
        },
      },
    };

    const supportsTerminalAuth = request.clientCapabilities?.auth?.terminal === true;
    const supportsMetaTerminalAuth = request.clientCapabilities?._meta?.["terminal-auth"] === true;

    const claudeLoginMethod: any = {
      description: "Use Claude subscription ",
      name: "Claude Subscription",
      id: "claude-ai-login",
      type: "terminal",
      args: ["--cli", "auth", "login", "--claudeai"],
    };

    const consoleLoginMethod: any = {
      description: "Use Anthropic Console (API usage billing)",
      name: "Anthropic Console",
      id: "console-login",
      type: "terminal",
      args: ["--cli", "auth", "login", "--console"],
    };

    // If client supports terminal-auth capability, use that instead.
    if (supportsMetaTerminalAuth) {
      const baseArgs = process.argv.slice(1);
      claudeLoginMethod._meta = {
        "terminal-auth": {
          command: process.execPath,
          args: [...baseArgs, "--cli", "auth", "login", "--claudeai"],
          label: "Claude Login",
        },
      };
      consoleLoginMethod._meta = {
        "terminal-auth": {
          command: process.execPath,
          args: [...baseArgs, "--cli", "auth", "login", "--console"],
          label: "Anthropic Console Login",
        },
      };
    }

    return {
      protocolVersion: 1,
      agentCapabilities: {
        _meta: {
          claudeCode: {
            promptQueueing: true,
          },
        },
        promptCapabilities: {
          image: true,
          embeddedContext: true,
        },
        mcpCapabilities: {
          http: true,
          sse: true,
        },
        loadSession: true,
        sessionCapabilities: {
          fork: {},
          list: {},
          resume: {},
          close: {},
        },
      },
      agentInfo: {
        name: packageJson.name,
        title: "Claude Agent",
        version: packageJson.version,
      },
      authMethods: [
        ...(!shouldHideClaudeAuth() && (supportsTerminalAuth || supportsMetaTerminalAuth)
          ? [claudeLoginMethod]
          : []),
        ...(supportsTerminalAuth || supportsMetaTerminalAuth ? [consoleLoginMethod] : []),
        ...(supportsGatewayAuth ? [gatewayAuthMethod] : []),
      ],
    };
  }

  async newSession(params: NewSessionRequest): Promise<NewSessionResponse> {
    if (
      !this.gatewayAuthMeta &&
      fs.existsSync(path.resolve(os.homedir(), ".claude.json.backup")) &&
      !fs.existsSync(path.resolve(os.homedir(), ".claude.json"))
    ) {
      throw RequestError.authRequired();
    }

    const response = await this.createSession(params, {
      // Revisit these meta values once we support resume
      resume: (params._meta as NewSessionMeta | undefined)?.claudeCode?.options?.resume,
    });
    // Needs to happen after we return the session
    setTimeout(() => {
      this.sendAvailableCommandsUpdate(response.sessionId);
    }, 0);
    return response;
  }

  async unstable_forkSession(params: ForkSessionRequest): Promise<ForkSessionResponse> {
    const response = await this.createSession(
      {
        cwd: params.cwd,
        mcpServers: params.mcpServers ?? [],
        _meta: params._meta,
      },
      {
        resume: params.sessionId,
        forkSession: true,
      },
    );
    // Needs to happen after we return the session
    setTimeout(() => {
      this.sendAvailableCommandsUpdate(response.sessionId);
    }, 0);
    return response;
  }

  async unstable_resumeSession(params: ResumeSessionRequest): Promise<ResumeSessionResponse> {
    const result = await this.getOrCreateSession(params);

    // Delay commands update to ensure SDK is fully initialized (#391).
    // setTimeout(fn, 0) can race with SDK init; 500ms gives it time
    // to load skills, commands, and plugins for the resumed session.
    setTimeout(() => {
      this.sendAvailableCommandsUpdate(params.sessionId);
    }, 500);
    return result;
  }

  async loadSession(params: LoadSessionRequest): Promise<LoadSessionResponse> {
    const result = await this.getOrCreateSession(params);

    await this.replaySessionHistory(params.sessionId);

    // Send available commands after replay so it doesn't interleave with history
    setTimeout(() => {
      this.sendAvailableCommandsUpdate(params.sessionId);
    }, 0);

    return result;
  }

  async listSessions(params: ListSessionsRequest): Promise<ListSessionsResponse> {
    const sdk_sessions = await listSessions({ dir: params.cwd ?? undefined });
    const sessions = [];

    for (const session of sdk_sessions) {
      if (!session.cwd) continue;
      sessions.push({
        sessionId: session.sessionId,
        cwd: session.cwd,
        title: sanitizeTitle(session.summary),
        updatedAt: new Date(session.lastModified).toISOString(),
      });
    }
    return {
      sessions,
    };
  }

  async authenticate(_params: AuthenticateRequest): Promise<void> {
    if (_params.methodId === "gateway") {
      this.gatewayAuthMeta = _params._meta as GatewayAuthMeta | undefined;
      return;
    }
    throw new Error("Method not implemented.");
  }

  async prompt(params: PromptRequest): Promise<PromptResponse> {
    const session = this.sessions[params.sessionId];
    if (!session) {
      throw new Error("Session not found");
    }

    session.cancelled = false;
    session.accumulatedUsage = {
      inputTokens: 0,
      outputTokens: 0,
      cachedReadTokens: 0,
      cachedWriteTokens: 0,
    };

    let lastAssistantTotalUsage: number | null = null;
    let lastAssistantModel: string | null = null;
    let lastContextWindowSize: number = 200000;

    const userMessage = promptToClaude(params);

    const promptUuid = randomUUID();
    userMessage.uuid = promptUuid;

    // These local-only commands return a result without replaying the user
    // message. Mark promptReplayed=true so their result isn't consumed as a
    // background task result.
    const firstText = params.prompt[0]?.type === "text" ? params.prompt[0].text : "";
    const isLocalOnlyCommand =
      firstText.startsWith("/") && LOCAL_ONLY_COMMANDS.has(firstText.split(" ", 1)[0]);

    if (session.promptRunning) {
      session.input.push(userMessage);
      const order = session.nextPendingOrder++;
      const cancelled = await new Promise<boolean>((resolve) => {
        session.pendingMessages.set(promptUuid, { resolve, order });
      });
      if (cancelled) {
        return { stopReason: "cancelled" };
      }
    } else {
      session.input.push(userMessage);
    }

    session.promptRunning = true;
    let handedOff = false;
    let stopReason: StopReason = "end_turn";
    let gotResult = false;
    let resultTimeout: ReturnType<typeof setTimeout> | null = null;

    try {
      while (true) {
        // If we got a result but idle never came, wait up to 5s then return
        const nextPromise = session.query.next();
        let timedOut = false;

        const { value: message, done } = gotResult
          ? await Promise.race([
              nextPromise,
              new Promise<{ value: undefined; done: true }>((resolve) => {
                resultTimeout = setTimeout(() => {
                  timedOut = true;
                  resolve({ value: undefined, done: true });
                }, 5000);
              }),
            ])
          : await nextPromise;

        if (resultTimeout) {
          clearTimeout(resultTimeout);
          resultTimeout = null;
        }

        if (timedOut) {
          this.logger.log("Idle state not received after result, returning with timeout fallback");
          return { stopReason, usage: sessionUsage(session) };
        }

        if (done || !message) {
          if (session.cancelled) {
            return { stopReason: "cancelled" };
          }
          if (gotResult) {
            return { stopReason, usage: sessionUsage(session) };
          }
          break;
        }

        switch (message.type) {
          case "system":
            switch (message.subtype) {
              case "init":
                break;
              case "status": {
                if (message.status === "compacting") {
                  await this.client.sessionUpdate({
                    sessionId: message.session_id,
                    update: {
                      sessionUpdate: "agent_message_chunk",
                      content: { type: "text", text: "Compacting..." },
                    },
                  });
                }
                break;
              }
              case "compact_boundary": {
                // Send used:0 immediately so the client doesn't keep showing
                // the stale pre-compaction context size until the next turn.
                //
                // This is a deliberate approximation: we don't know the exact
                // post-compaction token count (only the SDK's next API call
                // reveals that). But used:0 is directionally correct — context
                // just dropped dramatically — and the real value replaces it
                // within seconds when the next result message arrives.
                // The alternative (no update) leaves the client showing e.g.
                // "944k/1m" right after the user sees "Compacting completed",
                // which is confusing and wrong.
                lastAssistantTotalUsage = 0;
                await this.client.sessionUpdate({
                  sessionId: message.session_id,
                  update: {
                    sessionUpdate: "usage_update",
                    used: 0,
                    size: lastContextWindowSize,
                  },
                });
                await this.client.sessionUpdate({
                  sessionId: message.session_id,
                  update: {
                    sessionUpdate: "agent_message_chunk",
                    content: { type: "text", text: "\n\nCompacting completed." },
                  },
                });
                break;
              }
              case "local_command_output": {
                await this.client.sessionUpdate({
                  sessionId: message.session_id,
                  update: {
                    sessionUpdate: "agent_message_chunk",
                    content: { type: "text", text: message.content },
                  },
                });
                // Local commands don't always emit idle state (#435).
                // Mark gotResult so the timeout fallback kicks in quickly.
                if (isLocalOnlyCommand) {
                  gotResult = true;
                }
                break;
              }
              case "session_state_changed": {
                if (message.state === "idle") {
                  return { stopReason, usage: sessionUsage(session) };
                }
                break;
              }
              case "hook_started":
              case "hook_progress":
              case "hook_response":
              case "files_persisted":
              case "elicitation_complete":
              case "api_retry":
                // Todo: process via status api
                break;
              case "task_started": {
                // Forward background task start to client (#336)
                const taskMsg = message as any;
                await this.client.sessionUpdate({
                  sessionId: taskMsg.session_id ?? params.sessionId,
                  update: {
                    sessionUpdate: "agent_message_chunk",
                    content: { type: "text", text: `\n🔄 Background task started: ${taskMsg.task_id ?? "unknown"}\n` },
                  },
                });
                break;
              }
              case "task_notification": {
                // Forward task notifications to client (#336)
                const taskNotif = message as any;
                if (taskNotif.message || taskNotif.content) {
                  await this.client.sessionUpdate({
                    sessionId: taskNotif.session_id ?? params.sessionId,
                    update: {
                      sessionUpdate: "agent_message_chunk",
                      content: { type: "text", text: taskNotif.message ?? taskNotif.content ?? "" },
                    },
                  });
                }
                break;
              }
              case "task_progress": {
                // Forward task progress to client (#336)
                const taskProg = message as any;
                if (taskProg.content) {
                  await this.client.sessionUpdate({
                    sessionId: taskProg.session_id ?? params.sessionId,
                    update: {
                      sessionUpdate: "agent_message_chunk",
                      content: { type: "text", text: taskProg.content },
                    },
                  });
                }
                break;
              }
              default:
                unreachable(message, this.logger);
                break;
            }
            break;
          case "result": {
            // Accumulate usage from this result
            session.accumulatedUsage.inputTokens += message.usage.input_tokens;
            session.accumulatedUsage.outputTokens += message.usage.output_tokens;
            session.accumulatedUsage.cachedReadTokens += message.usage.cache_read_input_tokens;
            session.accumulatedUsage.cachedWriteTokens += message.usage.cache_creation_input_tokens;

            const matchingModelUsage = lastAssistantModel
              ? getMatchingModelUsage(message.modelUsage, lastAssistantModel)
              : null;
            const contextWindowSize = matchingModelUsage?.contextWindow ?? 200000;
            lastContextWindowSize = contextWindowSize;

            // Send usage_update notification
            // Guard against null/undefined used value (#375)
            const usedTokens = lastAssistantTotalUsage ?? 0;
            await this.client.sessionUpdate({
              sessionId: params.sessionId,
              update: {
                sessionUpdate: "usage_update",
                used: usedTokens,
                size: contextWindowSize,
                cost: {
                  amount: message.total_cost_usd ?? 0,
                  currency: "USD",
                },
              },
            });

            if (session.cancelled) {
              stopReason = "cancelled";
              break;
            }

            switch (message.subtype) {
              case "success": {
                if (message.result.includes("Please run /login")) {
                  throw RequestError.authRequired();
                }
                if (message.stop_reason === "max_tokens") {
                  stopReason = "max_tokens";
                  break;
                }
                if (message.is_error) {
                  throw RequestError.internalError(undefined, message.result);
                }
                // Forward result text when no stream events were emitted
                // (fixes #453: output_tokens=0 means no stream_event chunks,
                // so the result text is the only content to show)
                if (isLocalOnlyCommand || message.usage.output_tokens === 0) {
                  if (message.result) {
                    for (const notification of toAcpNotifications(
                      message.result,
                      "assistant",
                      params.sessionId,
                      this.toolUseCache,
                      this.client,
                      this.logger,
                    )) {
                      await this.client.sessionUpdate(notification);
                    }
                  }
                }
                break;
              }
              case "error_during_execution": {
                if (message.stop_reason === "max_tokens") {
                  stopReason = "max_tokens";
                  break;
                }
                if (message.is_error) {
                  throw RequestError.internalError(
                    undefined,
                    message.errors.join(", ") || message.subtype,
                  );
                }
                stopReason = "end_turn";
                break;
              }
              case "error_max_budget_usd":
              case "error_max_turns":
              case "error_max_structured_output_retries":
                if (message.is_error) {
                  throw RequestError.internalError(
                    undefined,
                    message.errors.join(", ") || message.subtype,
                  );
                }
                stopReason = "max_turn_requests";
                break;
              default:
                unreachable(message, this.logger);
                break;
            }
            // Mark that we got a result — start idle timeout fallback
            gotResult = true;
            break;
          }
          case "stream_event": {
            for (const notification of streamEventToAcpNotifications(
              message,
              params.sessionId,
              this.toolUseCache,
              this.client,
              this.logger,
              {
                clientCapabilities: this.clientCapabilities,
                cwd: session.cwd,
              },
            )) {
              await this.client.sessionUpdate(notification);
            }
            break;
          }
          case "user":
          case "assistant": {
            if (session.cancelled) {
              break;
            }

            // Check for prompt replay
            if (message.type === "user" && "uuid" in message && message.uuid) {
              if (message.uuid === promptUuid) {
                break;
              }

              const pending = session.pendingMessages.get(message.uuid as string);
              if (pending) {
                pending.resolve(false);
                session.pendingMessages.delete(message.uuid as string);
                handedOff = true;
                // the current loop stops with end_turn,
                // the loop of the next prompt continues running
                return { stopReason: "end_turn", usage: sessionUsage(session) };
              }
              if ("isReplay" in message && message.isReplay) {
                // not pending or unrelated replay message
                break;
              }
            }

            // Store latest assistant usage (excluding subagents)
            // Sum all token types as a proxy for post-turn context occupancy:
            // current turn's output will become next turn's input.
            // Note: per the Anthropic API, input_tokens excludes cache tokens —
            // cache_read and cache_creation are reported separately, so summing
            // all four fields is not double-counting.
            if ((message.message as any).usage && message.parent_tool_use_id === null) {
              const messageWithUsage = message.message as unknown as SDKResultMessage;
              lastAssistantTotalUsage =
                messageWithUsage.usage.input_tokens +
                messageWithUsage.usage.output_tokens +
                messageWithUsage.usage.cache_read_input_tokens +
                messageWithUsage.usage.cache_creation_input_tokens;
            }
            // Track the current top-level model for context window size lookup
            // (exclude subagent messages to stay in sync with lastAssistantTotalUsage)
            if (
              message.type === "assistant" &&
              message.parent_tool_use_id === null &&
              message.message.model &&
              message.message.model !== "<synthetic>"
            ) {
              lastAssistantModel = message.message.model;
            }

            // Slash commands like /compact can generate invalid output... doesn't match
            // their own docs: https://docs.anthropic.com/en/docs/claude-code/sdk/sdk-slash-commands#%2Fcompact-compact-conversation-history
            if (
              typeof message.message.content === "string" &&
              message.message.content.includes("<local-command-stdout>")
            ) {
              this.logger.log(message.message.content);
              break;
            }

            if (
              typeof message.message.content === "string" &&
              message.message.content.includes("<local-command-stderr>")
            ) {
              this.logger.error(message.message.content);
              break;
            }
            // Skip these user messages for now, since they seem to just be messages we don't want in the feed
            if (
              message.type === "user" &&
              (typeof message.message.content === "string" ||
                (Array.isArray(message.message.content) &&
                  message.message.content.length === 1 &&
                  message.message.content[0].type === "text"))
            ) {
              break;
            }

            if (
              message.type === "assistant" &&
              message.message.model === "<synthetic>" &&
              Array.isArray(message.message.content) &&
              message.message.content.length === 1 &&
              message.message.content[0].type === "text" &&
              message.message.content[0].text.includes("Please run /login")
            ) {
              throw RequestError.authRequired();
            }

            const content =
              message.type === "assistant"
                ? // Handled by stream events above
                  message.message.content.filter(
                    (item) => !["text", "thinking"].includes(item.type),
                  )
                : message.message.content;

            for (const notification of toAcpNotifications(
              content,
              message.message.role,
              params.sessionId,
              this.toolUseCache,
              this.client,
              this.logger,
              {
                clientCapabilities: this.clientCapabilities,
                parentToolUseId: message.parent_tool_use_id,
                cwd: session.cwd,
              },
            )) {
              await this.client.sessionUpdate(notification);
            }
            break;
          }
          case "tool_progress":
          case "tool_use_summary":
          case "auth_status":
          case "prompt_suggestion":
          case "rate_limit_event":
            break;
          default:
            unreachable(message);
            break;
        }
      }
      // If loop ended without result (query iterator exhausted),
      // return gracefully instead of crashing (#497)
      this.logger.log("Session query iterator ended without explicit result");
      return { stopReason, usage: sessionUsage(session) };
    } catch (error) {
      if (error instanceof RequestError || !(error instanceof Error)) {
        throw error;
      }
      const message = error.message;
      if (
        message.includes("ProcessTransport") ||
        message.includes("terminated process") ||
        message.includes("process exited with") ||
        message.includes("process terminated by signal") ||
        message.includes("Failed to write to process stdin")
      ) {
        this.logger.error(`Session ${params.sessionId}: Claude Agent process died: ${message}`);
        session.settingsManager.dispose();
        session.input.end();
        delete this.sessions[params.sessionId];
        throw RequestError.internalError(
          undefined,
          "The Claude Agent process exited unexpectedly. Please start a new session.",
        );
      }
      throw error;
    } finally {
      if (!handedOff) {
        session.promptRunning = false;
        // Resolve ALL pending prompt calls to ensure none get stuck (#283).
        // Previously only the first pending was resolved, leaving others
        // hanging forever if Claude didn't replay their messages.
        if (session.pendingMessages.size > 0) {
          const sorted = [...session.pendingMessages.entries()].sort(
            (a, b) => a[1].order - b[1].order,
          );
          for (const [uuid, pending] of sorted) {
            pending.resolve(false);
            session.pendingMessages.delete(uuid);
          }
        }
      }
    }
  }

  async cancel(params: CancelNotification): Promise<void> {
    const session = this.sessions[params.sessionId];
    if (!session) {
      throw new Error("Session not found");
    }
    session.cancelled = true;
    for (const [, pending] of session.pendingMessages) {
      pending.resolve(true);
    }
    session.pendingMessages.clear();
    await session.query.interrupt();
  }

  async unstable_closeSession(params: CloseSessionRequest): Promise<CloseSessionResponse> {
    const session = this.sessions[params.sessionId];
    if (!session) {
      throw new Error("Session not found");
    }
    await this.cancel({ sessionId: params.sessionId });

    session.settingsManager.dispose();
    session.abortController.abort();
    delete this.sessions[params.sessionId];

    return {};
  }

  async unstable_setSessionModel(
    params: SetSessionModelRequest,
  ): Promise<SetSessionModelResponse | void> {
    if (!this.sessions[params.sessionId]) {
      throw new Error("Session not found");
    }
    await this.sessions[params.sessionId].query.setModel(params.modelId);
    await this.updateConfigOption(params.sessionId, "model", params.modelId);
  }

  async setSessionMode(params: SetSessionModeRequest): Promise<SetSessionModeResponse> {
    if (!this.sessions[params.sessionId]) {
      throw new Error("Session not found");
    }

    await this.applySessionMode(params.sessionId, params.modeId);
    await this.updateConfigOption(params.sessionId, "mode", params.modeId);
    return {};
  }

  async setSessionConfigOption(
    params: SetSessionConfigOptionRequest,
  ): Promise<SetSessionConfigOptionResponse> {
    const session = this.sessions[params.sessionId];
    if (!session) {
      throw new Error("Session not found");
    }
    if (typeof params.value !== "string") {
      throw new Error(`Invalid value for config option ${params.configId}: ${params.value}`);
    }

    const option = session.configOptions.find((o) => o.id === params.configId);
    if (!option) {
      throw new Error(`Unknown config option: ${params.configId}`);
    }

    const allValues =
      "options" in option && Array.isArray(option.options)
        ? option.options.flatMap((o) => ("options" in o ? o.options : [o]))
        : [];
    let validValue = allValues.find((o) => o.value === params.value);

    // For model options, fall back to resolveModelPreference when the exact
    // value doesn't match.  This lets callers use human-friendly aliases like
    // "opus" or "sonnet" instead of full model IDs like "claude-opus-4-6".
    if (!validValue && params.configId === "model") {
      const modelInfos: ModelInfo[] = allValues.map((o) => ({
        value: o.value,
        displayName: o.name,
        description: o.description ?? "",
      }));
      const resolved = resolveModelPreference(modelInfos, params.value);
      if (resolved) {
        validValue = allValues.find((o) => o.value === resolved.value);
      }
    }

    if (!validValue) {
      throw new Error(`Invalid value for config option ${params.configId}: ${params.value}`);
    }

    // Use the canonical option value so downstream code always receives the
    // model ID rather than the caller-supplied alias.
    const resolvedValue = validValue.value;

    if (params.configId === "mode") {
      await this.applySessionMode(params.sessionId, resolvedValue);
      await this.client.sessionUpdate({
        sessionId: params.sessionId,
        update: {
          sessionUpdate: "current_mode_update",
          currentModeId: resolvedValue,
        },
      });
    } else if (params.configId === "model") {
      await this.sessions[params.sessionId].query.setModel(resolvedValue);
    }

    this.syncSessionConfigState(session, params.configId, params.value);

    session.configOptions = session.configOptions.map((o) =>
      o.id === params.configId && typeof o.currentValue === "string"
        ? { ...o, currentValue: resolvedValue }
        : o,
    );

    return { configOptions: session.configOptions };
  }

  private async applySessionMode(sessionId: string, modeId: string): Promise<void> {
    switch (modeId) {
      case "default":
      case "acceptEdits":
      case "bypassPermissions":
      case "dontAsk":
      case "plan":
      case "auto":
        break;
      default:
        throw new Error(`Invalid Mode: ${modeId}`);
    }
    try {
      await this.sessions[sessionId].query.setPermissionMode(modeId);
    } catch (error) {
      if (error instanceof Error) {
        if (!error.message) {
          error.message = "Invalid Mode";
        }
        throw error;
      } else {
        // eslint-disable-next-line preserve-caught-error
        throw new Error("Invalid Mode");
      }
    }
  }

  private async replaySessionHistory(sessionId: string): Promise<void> {
    const toolUseCache: ToolUseCache = {};
    const messages = await getSessionMessages(sessionId);

    for (const message of messages) {
      for (const notification of toAcpNotifications(
        // @ts-expect-error - untyped in SDK but we handle all of these
        message.message.content,
        // @ts-expect-error - untyped in SDK but we handle all of these
        message.message.role,
        sessionId,
        toolUseCache,
        this.client,
        this.logger,
        {
          registerHooks: false,
          clientCapabilities: this.clientCapabilities,
          cwd: this.sessions[sessionId]?.cwd,
        },
      )) {
        await this.client.sessionUpdate(notification);
      }
    }
  }

  async readTextFile(params: ReadTextFileRequest): Promise<ReadTextFileResponse> {
    const response = await this.client.readTextFile(params);
    return response;
  }

  async writeTextFile(params: WriteTextFileRequest): Promise<WriteTextFileResponse> {
    const response = await this.client.writeTextFile(params);
    return response;
  }

  canUseTool(sessionId: string): CanUseTool {
    return async (toolName, toolInput, { signal, suggestions, toolUseID }) => {
      const supportsTerminalOutput = this.clientCapabilities?._meta?.["terminal_output"] === true;
      const session = this.sessions[sessionId];
      if (!session) {
        return {
          behavior: "deny",
          message: "Session not found",
        };
      }

      if (toolName === "ExitPlanMode") {
        const options = [
          {
            kind: "allow_always",
            name: "Yes, and auto-accept edits",
            optionId: "acceptEdits",
          },
          { kind: "allow_once", name: "Yes, and manually approve edits", optionId: "default" },
          { kind: "reject_once", name: "No, keep planning", optionId: "plan" },
        ];
        if (ALLOW_BYPASS) {
          options.unshift({
            kind: "allow_always",
            name: "Yes, and bypass permissions",
            optionId: "bypassPermissions",
          });
        }

        const response = await this.client.requestPermission({
          options,
          sessionId,
          toolCall: {
            toolCallId: toolUseID,
            rawInput: toolInput,
            ...toolInfoFromToolUse(
              { name: toolName, input: toolInput, id: toolUseID },
              supportsTerminalOutput,
              session?.cwd,
            ),
          },
        });

        if (signal.aborted || response.outcome?.outcome === "cancelled") {
          throw new Error("Tool use aborted");
        }
        if (
          response.outcome?.outcome === "selected" &&
          (response.outcome.optionId === "default" ||
            response.outcome.optionId === "acceptEdits" ||
            response.outcome.optionId === "bypassPermissions")
        ) {
          await this.client.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "current_mode_update",
              currentModeId: response.outcome.optionId,
            },
          });
          await this.updateConfigOption(sessionId, "mode", response.outcome.optionId);

          return {
            behavior: "allow",
            updatedInput: toolInput,
            updatedPermissions: suggestions ?? [
              { type: "setMode", mode: response.outcome.optionId, destination: "session" },
            ],
          };
        } else {
          return {
            behavior: "deny",
            message: "User rejected request to exit plan mode.",
          };
        }
      }

      if (session.modes.currentModeId === "bypassPermissions") {
        return {
          behavior: "allow",
          updatedInput: toolInput,
          updatedPermissions: suggestions ?? [
            { type: "addRules", rules: [{ toolName }], behavior: "allow", destination: "session" },
          ],
        };
      }

      // Request permission with timeout fallback (#329)
      // Subagent tool calls may not surface permission prompts in UI,
      // causing silent hangs. Timeout after 60s and deny.
      const PERMISSION_TIMEOUT = 60_000;
      let response: Awaited<ReturnType<typeof this.client.requestPermission>>;
      try {
        response = await Promise.race([
          this.client.requestPermission({
            options: [
              {
                kind: "allow_always",
                name: "Always Allow",
                optionId: "allow_always",
              },
              { kind: "allow_once", name: "Allow", optionId: "allow" },
              { kind: "reject_once", name: "Reject", optionId: "reject" },
            ],
            sessionId,
            toolCall: {
              toolCallId: toolUseID,
              rawInput: toolInput,
              ...toolInfoFromToolUse(
                { name: toolName, input: toolInput, id: toolUseID },
                supportsTerminalOutput,
                session?.cwd,
              ),
            },
          }),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error("Permission request timed out")), PERMISSION_TIMEOUT)
          ),
        ]);
      } catch (e) {
        this.logger.error(`Permission request for ${toolName} timed out or failed: ${e}`);
        return {
          behavior: "deny" as const,
          message: `Permission request timed out for ${toolName}. Check if the permission prompt was shown in the UI.`,
        };
      }
      if (signal.aborted || response.outcome?.outcome === "cancelled") {
        throw new Error("Tool use aborted");
      }
      if (
        response.outcome?.outcome === "selected" &&
        (response.outcome.optionId === "allow" || response.outcome.optionId === "allow_always")
      ) {
        // If Claude Code has suggestions, it will update their settings already
        if (response.outcome.optionId === "allow_always") {
          return {
            behavior: "allow",
            updatedInput: toolInput,
            updatedPermissions: suggestions ?? [
              {
                type: "addRules",
                rules: [{ toolName }],
                behavior: "allow",
                destination: "session",
              },
            ],
          };
        }
        return {
          behavior: "allow",
          updatedInput: toolInput,
        };
      } else {
        return {
          behavior: "deny",
          message: "User refused permission to run tool",
        };
      }
    };
  }

  /**
   * Discover installed Claude Code plugins (#461).
   * Scans ~/.claude/plugins/ for directories containing .claude-plugin/plugin.json
   */
  private discoverPlugins(): string[] {
    const pluginDirs: string[] = [];
    try {
      const baseDir = path.join(CLAUDE_CONFIG_DIR, "plugins");
      if (!fs.existsSync(baseDir)) return pluginDirs;

      // Scan direct children and marketplace subdirs
      const scanDir = (dir: string) => {
        for (const entry of fs.readdirSync(dir)) {
          const full = path.join(dir, entry);
          if (!fs.statSync(full).isDirectory()) continue;
          // Check if this directory is a plugin (has .claude-plugin/plugin.json)
          const manifest = path.join(full, ".claude-plugin", "plugin.json");
          if (fs.existsSync(manifest)) {
            pluginDirs.push(full);
          }
          // Check one level deeper (for marketplaces/org/plugin structure)
          if (entry === "marketplaces" || entry === "cache") {
            for (const subEntry of fs.readdirSync(full)) {
              const subFull = path.join(full, subEntry);
              if (fs.statSync(subFull).isDirectory()) {
                scanDir(subFull);
              }
            }
          }
        }
      };
      scanDir(baseDir);
      if (pluginDirs.length > 0) {
        this.logger.log(`Discovered ${pluginDirs.length} plugins: ${pluginDirs.map(p => path.basename(p)).join(", ")}`);
      }
    } catch (e) {
      this.logger.error(`Plugin discovery failed: ${e}`);
    }
    return pluginDirs;
  }

  private async sendAvailableCommandsUpdate(sessionId: string): Promise<void> {
    const session = this.sessions[sessionId];
    if (!session) return;
    const commands = await session.query.supportedCommands();
    await this.client.sessionUpdate({
      sessionId,
      update: {
        sessionUpdate: "available_commands_update",
        availableCommands: getAvailableSlashCommands(commands),
      },
    });
  }

  private async updateConfigOption(
    sessionId: string,
    configId: string,
    value: string,
  ): Promise<void> {
    const session = this.sessions[sessionId];
    if (!session) return;

    this.syncSessionConfigState(session, configId, value);

    session.configOptions = session.configOptions.map((o) =>
      o.id === configId && typeof o.currentValue === "string" ? { ...o, currentValue: value } : o,
    );

    await this.client.sessionUpdate({
      sessionId,
      update: {
        sessionUpdate: "config_option_update",
        configOptions: session.configOptions,
      },
    });
  }

  private syncSessionConfigState(session: Session, configId: string, value: string): void {
    if (configId === "mode") {
      session.modes = { ...session.modes, currentModeId: value };
    } else if (configId === "model") {
      session.models = { ...session.models, currentModelId: value };
    }
  }

  private async getOrCreateSession(params: {
    sessionId: string;
    cwd: string;
    mcpServers?: NewSessionRequest["mcpServers"];
    _meta?: NewSessionRequest["_meta"];
  }): Promise<NewSessionResponse> {
    const existingSession = this.sessions[params.sessionId];
    if (existingSession) {
      return {
        sessionId: params.sessionId,
        modes: existingSession.modes,
        models: existingSession.models,
        configOptions: existingSession.configOptions,
      };
    }

    const response = await this.createSession(
      {
        cwd: params.cwd,
        mcpServers: params.mcpServers ?? [],
        _meta: params._meta,
      },
      {
        resume: params.sessionId,
      },
    );

    return {
      sessionId: response.sessionId,
      modes: response.modes,
      models: response.models,
      configOptions: response.configOptions,
    };
  }

  private async createSession(
    params: NewSessionRequest,
    creationOpts: { resume?: string; forkSession?: boolean } = {},
  ): Promise<NewSessionResponse> {
    // We want to create a new session id unless it is resume,
    // but not resume + forkSession.
    let sessionId;
    if (creationOpts.forkSession) {
      sessionId = randomUUID();
    } else if (creationOpts.resume) {
      sessionId = creationOpts.resume;
    } else {
      sessionId = randomUUID();
    }

    const input = new Pushable<SDKUserMessage>();

    const settingsManager = new SettingsManager(params.cwd, {
      logger: this.logger,
    });
    await settingsManager.initialize();

    const mcpServers: Record<string, McpServerConfig> = {};
    if (Array.isArray(params.mcpServers)) {
      for (const server of params.mcpServers) {
        if ("type" in server && (server.type === "http" || server.type === "sse")) {
          // HTTP or SSE type MCP server
          mcpServers[server.name] = {
            type: server.type,
            url: server.url,
            headers: server.headers
              ? Object.fromEntries(server.headers.map((e) => [e.name, e.value]))
              : undefined,
          };
        } else {
          // Stdio type MCP server (with or without explicit type field)
          mcpServers[server.name] = {
            type: "stdio",
            command: server.command,
            args: server.args,
            env: server.env
              ? Object.fromEntries(server.env.map((e) => [e.name, e.value]))
              : undefined,
          };
        }
      }
    }

    let systemPrompt: Options["systemPrompt"] = { type: "preset", preset: "claude_code" };
    if (params._meta?.systemPrompt) {
      const customPrompt = params._meta.systemPrompt;
      if (typeof customPrompt === "string") {
        systemPrompt = customPrompt;
      } else if (
        typeof customPrompt === "object" &&
        "append" in customPrompt &&
        typeof customPrompt.append === "string"
      ) {
        systemPrompt.append = customPrompt.append;
      }
    }

    const permissionMode = resolvePermissionMode(
      settingsManager.getSettings().permissions?.defaultMode,
    );

    // Extract options from _meta if provided
    const sessionMeta = params._meta as NewSessionMeta | undefined;
    const userProvidedOptions = sessionMeta?.claudeCode?.options;

    // Configure thinking tokens from environment variable
    const maxThinkingTokens = process.env.MAX_THINKING_TOKENS
      ? parseInt(process.env.MAX_THINKING_TOKENS, 10)
      : undefined;

    // Disable this for now, not a great way to expose this over ACP at the moment (in progress work so we can revisit)
    const disallowedTools = ["AskUserQuestion"];

    // Resolve which built-in tools to expose.
    // Explicit tools array from _meta.claudeCode.options takes precedence.
    // disableBuiltInTools is a legacy shorthand for tools: [] — kept for
    // backward compatibility but callers should prefer the tools array.
    const tools: Options["tools"] =
      userProvidedOptions?.tools ??
      (params._meta?.disableBuiltInTools === true ? [] : { type: "preset", preset: "claude_code" });

    const abortController = userProvidedOptions?.abortController || new AbortController();

    // Load installed plugins (#461) — scan ~/.claude/plugins/
    const pluginPaths = this.discoverPlugins();

    const options: Options = {
      systemPrompt,
      settingSources: ["user", "project", "local"],
      ...(pluginPaths.length > 0 && {
        plugins: pluginPaths.map((p) => ({ type: "local" as const, path: p })),
      }),
      ...(maxThinkingTokens !== undefined && { maxThinkingTokens }),
      ...userProvidedOptions,
      // Only set env if we have custom vars beyond the defaults (#404)
      // Setting env explicitly can break Bash tool's default env inheritance
      env: Object.keys(userProvidedOptions?.env ?? {}).length > 0 || this.gatewayAuthMeta
        ? {
            ...process.env,
            ...userProvidedOptions?.env,
            ...createEnvForGateway(this.gatewayAuthMeta),
            CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS: "1",
          }
        : {
            ...process.env,
            CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS: "1",
          },
      // Override certain fields that must be controlled by ACP
      cwd: params.cwd,
      includePartialMessages: true,
      mcpServers: { ...(userProvidedOptions?.mcpServers || {}), ...mcpServers },
      // If we want bypassPermissions to be an option, we have to allow it here.
      // But it doesn't work in root mode, so we only activate it if it will work.
      allowDangerouslySkipPermissions: ALLOW_BYPASS,
      permissionMode,
      canUseTool: this.canUseTool(sessionId),
      // note: although not documented by the types, passing an absolute path
      // here works to find zed's managed node version.
      executable: isStaticBinary() ? undefined : (process.execPath as any),
      ...(process.env.CLAUDE_CODE_EXECUTABLE
        ? { pathToClaudeCodeExecutable: process.env.CLAUDE_CODE_EXECUTABLE }
        : isStaticBinary()
          ? { pathToClaudeCodeExecutable: await claudeCliPath() }
          : {}),
      extraArgs: {
        ...userProvidedOptions?.extraArgs,
        "replay-user-messages": "",
      },
      disallowedTools: [...(userProvidedOptions?.disallowedTools || []), ...disallowedTools],
      tools,
      hooks: {
        ...userProvidedOptions?.hooks,
        PostToolUse: [
          ...(userProvidedOptions?.hooks?.PostToolUse || []),
          {
            hooks: [
              createPostToolUseHook(this.logger, {
                onEnterPlanMode: async () => {
                  await this.client.sessionUpdate({
                    sessionId,
                    update: {
                      sessionUpdate: "current_mode_update",
                      currentModeId: "plan",
                    },
                  });
                  await this.updateConfigOption(sessionId, "mode", "plan");
                },
              }),
            ],
          },
        ],
      },
      ...creationOpts,
      abortController,
    };

    options.additionalDirectories = [
      ...(userProvidedOptions?.additionalDirectories ?? []),
      ...(sessionMeta?.additionalRoots ?? []),
    ];

    if (creationOpts?.resume === undefined || creationOpts?.forkSession) {
      // Set our own session id if not resuming an existing session.
      options.sessionId = sessionId;
    }

    // Handle abort controller from meta options
    if (abortController?.signal.aborted) {
      throw new Error("Cancelled");
    }

    const q = query({
      prompt: input,
      options,
    });

    let initializationResult;
    try {
      initializationResult = await q.initializationResult();
    } catch (error) {
      if (
        creationOpts.resume &&
        error instanceof Error &&
        error.message === "Query closed before response received"
      ) {
        throw RequestError.resourceNotFound(sessionId);
      }
      throw error;
    }

    if (
      shouldHideClaudeAuth() &&
      initializationResult.account.subscriptionType &&
      !this.gatewayAuthMeta
    ) {
      throw RequestError.authRequired(
        undefined,
        "This integration does not support using claude.ai subscriptions.",
      );
    }

    const models = await getAvailableModels(q, initializationResult.models, settingsManager);

    const availableModes = [
      {
        id: "auto",
        name: "Auto",
        decription: "Use a model classifier to approve/deny permission prompts.",
      },
      {
        id: "default",
        name: "Default",
        description: "Standard behavior, prompts for dangerous operations",
      },
      {
        id: "acceptEdits",
        name: "Accept Edits",
        description: "Auto-accept file edit operations",
      },
      {
        id: "plan",
        name: "Plan Mode",
        description: "Planning mode, no actual tool execution",
      },
      {
        id: "dontAsk",
        name: "Don't Ask",
        description: "Don't prompt for permissions, deny if not pre-approved",
      },
    ];
    // Only works in non-root mode
    if (ALLOW_BYPASS) {
      availableModes.push({
        id: "bypassPermissions",
        name: "Bypass Permissions",
        description: "Bypass all permission checks",
      });
    }

    const modes = {
      currentModeId: permissionMode,
      availableModes,
    };

    const configOptions = buildConfigOptions(modes, models);

    this.sessions[sessionId] = {
      query: q,
      input: input,
      cancelled: false,
      cwd: params.cwd,
      settingsManager,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      modes,
      models,
      configOptions,
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController,
    };

    return {
      sessionId,
      models,
      modes,
      configOptions,
    };
  }
}

function sessionUsage(session: Session) {
  return {
    inputTokens: session.accumulatedUsage.inputTokens,
    outputTokens: session.accumulatedUsage.outputTokens,
    cachedReadTokens: session.accumulatedUsage.cachedReadTokens,
    cachedWriteTokens: session.accumulatedUsage.cachedWriteTokens,
    totalTokens:
      session.accumulatedUsage.inputTokens +
      session.accumulatedUsage.outputTokens +
      session.accumulatedUsage.cachedReadTokens +
      session.accumulatedUsage.cachedWriteTokens,
  };
}

function createEnvForGateway(gatewayMeta?: GatewayAuthMeta) {
  if (!gatewayMeta) {
    return {};
  }
  return {
    ANTHROPIC_BASE_URL: gatewayMeta.gateway.baseUrl,
    ANTHROPIC_CUSTOM_HEADERS: Object.entries(gatewayMeta.gateway.headers)
      .map(([key, value]) => `${key}: ${value}`)
      .join("\n"),
    ANTHROPIC_AUTH_TOKEN: "", // Must be specified to bypass claude login requirement
  };
}

function buildConfigOptions(
  modes: SessionModeState,
  models: SessionModelState,
): SessionConfigOption[] {
  return [
    {
      id: "mode",
      name: "Mode",
      description: "Session permission mode",
      category: "mode",
      type: "select",
      currentValue: modes.currentModeId,
      options: modes.availableModes.map((m) => ({
        value: m.id,
        name: m.name,
        description: m.description,
      })),
    },
    {
      id: "model",
      name: "Model",
      description: "AI model to use",
      category: "model",
      type: "select",
      currentValue: models.currentModelId,
      options: models.availableModels.map((m) => ({
        value: m.modelId,
        name: m.name,
        description: m.description ?? undefined,
      })),
    },
  ];
}

// Claude Code CLI persists display strings like "opus[1m]" in settings,
// but the SDK model list uses IDs like "claude-opus-4-6-1m".
const MODEL_CONTEXT_HINT_PATTERN = /\[(\d+m)\]$/i;

function tokenizeModelPreference(model: string): { tokens: string[]; contextHint?: string } {
  const lower = model.trim().toLowerCase();
  const contextHint = lower.match(MODEL_CONTEXT_HINT_PATTERN)?.[1]?.toLowerCase();

  const normalized = lower.replace(MODEL_CONTEXT_HINT_PATTERN, " $1 ");
  const rawTokens = normalized.split(/[^a-z0-9]+/).filter(Boolean);
  const tokens = rawTokens
    .map((token) => {
      if (token === "opusplan") return "opus";
      if (token === "best" || token === "default") return "";
      return token;
    })
    .filter((token) => token && token !== "claude")
    .filter((token) => /[a-z]/.test(token) || token.endsWith("m"));

  return { tokens, contextHint };
}

function scoreModelMatch(model: ModelInfo, tokens: string[], contextHint?: string): number {
  const haystack = `${model.value} ${model.displayName}`.toLowerCase();
  let score = 0;
  for (const token of tokens) {
    if (haystack.includes(token)) {
      score += token === contextHint ? 3 : 1;
    }
  }
  return score;
}

function resolveModelPreference(models: ModelInfo[], preference: string): ModelInfo | null {
  const trimmed = preference.trim();
  if (!trimmed) return null;

  const lower = trimmed.toLowerCase();

  // Exact match on value or display name
  const directMatch = models.find(
    (model) =>
      model.value === trimmed ||
      model.value.toLowerCase() === lower ||
      model.displayName.toLowerCase() === lower,
  );
  if (directMatch) return directMatch;

  // Substring match
  const includesMatch = models.find((model) => {
    const value = model.value.toLowerCase();
    const display = model.displayName.toLowerCase();
    return value.includes(lower) || display.includes(lower) || lower.includes(value);
  });
  if (includesMatch) return includesMatch;

  // Tokenized matching for aliases like "opus[1m]"
  const { tokens, contextHint } = tokenizeModelPreference(trimmed);
  if (tokens.length === 0) return null;

  let bestMatch: ModelInfo | null = null;
  let bestScore = 0;
  for (const model of models) {
    const score = scoreModelMatch(model, tokens, contextHint);
    if (0 < score && (!bestMatch || bestScore < score)) {
      bestMatch = model;
      bestScore = score;
    }
  }

  return bestMatch;
}

async function getAvailableModels(
  query: Query,
  models: ModelInfo[],
  settingsManager: SettingsManager,
): Promise<SessionModelState> {
  const settings = settingsManager.getSettings();

  let currentModel = models[0];

  if (settings.model) {
    const match = resolveModelPreference(models, settings.model);
    if (match) {
      currentModel = match;
    }
  }

  await query.setModel(currentModel.value);

  return {
    availableModels: models.map((model) => ({
      modelId: model.value,
      name: model.displayName,
      description: model.description,
    })),
    currentModelId: currentModel.value,
  };
}

function getAvailableSlashCommands(commands: SlashCommand[]): AvailableCommand[] {
  const UNSUPPORTED_COMMANDS = [
    "cost",
    "keybindings-help",
    "login",
    "logout",
    "output-style:new",
    "release-notes",
    "todos",
  ];

  return commands
    .map((command) => {
      const input = command.argumentHint
        ? {
            hint: Array.isArray(command.argumentHint)
              ? command.argumentHint.join(" ")
              : command.argumentHint,
          }
        : null;
      let name = command.name;
      if (command.name.endsWith(" (MCP)")) {
        name = `mcp:${name.replace(" (MCP)", "")}`;
      }
      return {
        name,
        description: command.description || "",
        input,
      };
    })
    .filter((command: AvailableCommand) => !UNSUPPORTED_COMMANDS.includes(command.name));
}

function formatUriAsLink(uri: string): string {
  try {
    if (uri.startsWith("file://")) {
      const path = uri.slice(7); // Remove "file://"
      const name = path.split("/").pop() || path;
      return `[@${name}](${uri})`;
    } else if (uri.startsWith("zed://")) {
      const parts = uri.split("/");
      const name = parts[parts.length - 1] || uri;
      return `[@${name}](${uri})`;
    }
    return uri;
  } catch {
    return uri;
  }
}

export function promptToClaude(prompt: PromptRequest): SDKUserMessage {
  const content: any[] = [];
  const context: any[] = [];

  for (const chunk of prompt.prompt) {
    switch (chunk.type) {
      case "text": {
        let text = chunk.text;
        // change /mcp:server:command args -> /server:command (MCP) args
        const mcpMatch = text.match(/^\/mcp:([^:\s]+):(\S+)(?:\s(.*))?$/);
        if (mcpMatch) {
          const [, server, command, args] = mcpMatch;
          text = `/${server}:${command} (MCP)${args ? ` ${args}` : ""}`;
        }
        content.push({ type: "text", text });
        break;
      }
      case "resource_link": {
        const formattedUri = formatUriAsLink(chunk.uri);
        content.push({
          type: "text",
          text: formattedUri,
        });
        break;
      }
      case "resource": {
        if ("text" in chunk.resource) {
          const formattedUri = formatUriAsLink(chunk.resource.uri);
          content.push({
            type: "text",
            text: formattedUri,
          });
          context.push({
            type: "text",
            text: `\n<context ref="${chunk.resource.uri}">\n${chunk.resource.text}\n</context>`,
          });
        }
        // Ignore blob resources (unsupported)
        break;
      }
      case "image":
        if (chunk.data) {
          content.push({
            type: "image",
            source: {
              type: "base64",
              data: chunk.data,
              media_type: chunk.mimeType,
            },
          });
        } else if (chunk.uri && chunk.uri.startsWith("http")) {
          content.push({
            type: "image",
            source: {
              type: "url",
              url: chunk.uri,
            },
          });
        }
        break;
      // Ignore audio and other unsupported types
      default:
        break;
    }
  }

  content.push(...context);

  return {
    type: "user",
    message: {
      role: "user",
      content: content,
    },
    session_id: prompt.sessionId,
    parent_tool_use_id: null,
  };
}

/**
 * Convert an SDKAssistantMessage (Claude) to a SessionNotification (ACP).
 * Only handles text, image, and thinking chunks for now.
 */
export function toAcpNotifications(
  content: string | ContentBlockParam[] | BetaContentBlock[] | BetaRawContentBlockDelta[],
  role: "assistant" | "user",
  sessionId: string,
  toolUseCache: ToolUseCache,
  client: AgentSideConnection,
  logger: Logger,
  options?: {
    registerHooks?: boolean;
    clientCapabilities?: ClientCapabilities;
    parentToolUseId?: string | null;
    cwd?: string;
  },
): SessionNotification[] {
  const registerHooks = options?.registerHooks !== false;
  const supportsTerminalOutput = options?.clientCapabilities?._meta?.["terminal_output"] === true;
  if (typeof content === "string") {
    const update: SessionNotification["update"] = {
      sessionUpdate: role === "assistant" ? "agent_message_chunk" : "user_message_chunk",
      content: {
        type: "text",
        text: content,
      },
    };

    if (options?.parentToolUseId) {
      update._meta = {
        ...update._meta,
        claudeCode: {
          ...(update._meta?.claudeCode || {}),
          parentToolUseId: options.parentToolUseId,
        },
      };
    }

    return [{ sessionId, update }];
  }

  const output = [];
  // Only handle the first chunk for streaming; extend as needed for batching
  for (const chunk of content) {
    let update: SessionNotification["update"] | null = null;
    switch (chunk.type) {
      case "text":
      case "text_delta":
        update = {
          sessionUpdate: role === "assistant" ? "agent_message_chunk" : "user_message_chunk",
          content: {
            type: "text",
            text: chunk.text,
          },
        };
        break;
      case "image":
        update = {
          sessionUpdate: role === "assistant" ? "agent_message_chunk" : "user_message_chunk",
          content: {
            type: "image",
            data: chunk.source.type === "base64" ? chunk.source.data : "",
            mimeType: chunk.source.type === "base64" ? chunk.source.media_type : "",
            uri: chunk.source.type === "url" ? chunk.source.url : undefined,
          },
        };
        break;
      case "thinking":
      case "thinking_delta":
        update = {
          sessionUpdate: "agent_thought_chunk",
          content: {
            type: "text",
            text: chunk.thinking,
          },
        };
        break;
      case "tool_use":
      case "server_tool_use":
      case "mcp_tool_use": {
        const alreadyCached = chunk.id in toolUseCache;
        toolUseCache[chunk.id] = chunk;
        if (chunk.name === "TodoWrite") {
          // @ts-expect-error - sometimes input is empty object
          if (Array.isArray(chunk.input.todos)) {
            update = {
              sessionUpdate: "plan",
              entries: planEntries(chunk.input as { todos: ClaudePlanEntry[] }),
            };
          }
        } else {
          // Only register hooks on first encounter to avoid double-firing
          if (registerHooks && !alreadyCached) {
            registerHookCallback(chunk.id, {
              onPostToolUseHook: async (toolUseId, toolInput, toolResponse) => {
                const toolUse = toolUseCache[toolUseId];
                if (toolUse) {
                  const editDiff =
                    toolUse.name === "Edit" ? toolUpdateFromEditToolResponse(toolResponse) : {};
                  const update: SessionNotification["update"] = {
                    _meta: {
                      claudeCode: {
                        toolResponse,
                        toolName: toolUse.name,
                      },
                    } satisfies ToolUpdateMeta,
                    toolCallId: toolUseId,
                    sessionUpdate: "tool_call_update",
                    ...editDiff,
                  };
                  await client.sessionUpdate({
                    sessionId,
                    update,
                  });
                } else {
                  logger.error(
                    `[claude-agent-acp] Got a tool response for tool use that wasn't tracked: ${toolUseId}`,
                  );
                }
              },
            });
          }

          let rawInput;
          try {
            rawInput = JSON.parse(JSON.stringify(chunk.input));
          } catch {
            // ignore if we can't turn it to JSON
          }

          if (alreadyCached) {
            // Second encounter (full assistant message after streaming) —
            // send as tool_call_update to refine the existing tool_call
            // rather than emitting a duplicate tool_call.
            update = {
              _meta: {
                claudeCode: {
                  toolName: chunk.name,
                },
              } satisfies ToolUpdateMeta,
              toolCallId: chunk.id,
              sessionUpdate: "tool_call_update",
              rawInput,
              ...toolInfoFromToolUse(chunk, supportsTerminalOutput, options?.cwd),
            };
          } else {
            // First encounter (streaming content_block_start or replay) —
            // send as tool_call with terminal_info for Bash tools.
            update = {
              _meta: {
                claudeCode: {
                  toolName: chunk.name,
                },
                ...(chunk.name === "Bash" && supportsTerminalOutput
                  ? { terminal_info: { terminal_id: chunk.id } }
                  : {}),
              } satisfies ToolUpdateMeta,
              toolCallId: chunk.id,
              sessionUpdate: "tool_call",
              rawInput,
              status: "pending",
              ...toolInfoFromToolUse(chunk, supportsTerminalOutput, options?.cwd),
            };
          }
        }
        break;
      }

      case "tool_result":
      case "tool_search_tool_result":
      case "web_fetch_tool_result":
      case "web_search_tool_result":
      case "code_execution_tool_result":
      case "bash_code_execution_tool_result":
      case "text_editor_code_execution_tool_result":
      case "mcp_tool_result": {
        const toolUse = toolUseCache[chunk.tool_use_id];
        if (!toolUse) {
          logger.error(
            `[claude-agent-acp] Got a tool result for tool use that wasn't tracked: ${chunk.tool_use_id}`,
          );
          break;
        }

        if (toolUse.name !== "TodoWrite") {
          const { _meta: toolMeta, ...toolUpdate } = toolUpdateFromToolResult(
            chunk,
            toolUseCache[chunk.tool_use_id],
            supportsTerminalOutput,
          );

          // When terminal output is supported, send terminal_output as a
          // separate notification to match codex-acp's streaming lifecycle:
          //   1. tool_call       → _meta.terminal_info  (already sent above)
          //   2. tool_call_update → _meta.terminal_output (sent here)
          //   3. tool_call_update → _meta.terminal_exit  (sent below with status)
          if (toolMeta?.terminal_output) {
            output.push({
              sessionId,
              update: {
                _meta: {
                  terminal_output: toolMeta.terminal_output,
                  ...(options?.parentToolUseId
                    ? { claudeCode: { parentToolUseId: options.parentToolUseId } }
                    : {}),
                },
                toolCallId: chunk.tool_use_id,
                sessionUpdate: "tool_call_update" as const,
              },
            });
          }

          update = {
            _meta: {
              claudeCode: {
                toolName: toolUse.name,
              },
              ...(toolMeta?.terminal_exit ? { terminal_exit: toolMeta.terminal_exit } : {}),
            } satisfies ToolUpdateMeta,
            toolCallId: chunk.tool_use_id,
            sessionUpdate: "tool_call_update",
            status: "is_error" in chunk && chunk.is_error ? "failed" : "completed",
            rawOutput: chunk.content,
            ...toolUpdate,
          };
        }
        break;
      }

      case "document":
      case "search_result":
      case "redacted_thinking":
      case "input_json_delta":
      case "citations_delta":
      case "signature_delta":
      case "container_upload":
      case "compaction":
      case "compaction_delta":
        break;

      default:
        unreachable(chunk, logger);
        break;
    }
    if (update) {
      if (options?.parentToolUseId) {
        update._meta = {
          ...update._meta,
          claudeCode: {
            ...(update._meta?.claudeCode || {}),
            parentToolUseId: options.parentToolUseId,
          },
        };
      }
      output.push({ sessionId, update });
    }
  }

  return output;
}

export function streamEventToAcpNotifications(
  message: SDKPartialAssistantMessage,
  sessionId: string,
  toolUseCache: ToolUseCache,
  client: AgentSideConnection,
  logger: Logger,
  options?: {
    clientCapabilities?: ClientCapabilities;
    cwd?: string;
  },
): SessionNotification[] {
  const event = message.event;
  switch (event.type) {
    case "content_block_start":
      return toAcpNotifications(
        [event.content_block],
        "assistant",
        sessionId,
        toolUseCache,
        client,
        logger,
        {
          clientCapabilities: options?.clientCapabilities,
          parentToolUseId: message.parent_tool_use_id,
          cwd: options?.cwd,
        },
      );
    case "content_block_delta":
      return toAcpNotifications(
        [event.delta],
        "assistant",
        sessionId,
        toolUseCache,
        client,
        logger,
        {
          clientCapabilities: options?.clientCapabilities,
          parentToolUseId: message.parent_tool_use_id,
          cwd: options?.cwd,
        },
      );
    // No content
    case "message_start":
    case "message_delta":
    case "message_stop":
    case "content_block_stop":
      return [];

    default:
      unreachable(event, logger);
      return [];
  }
}

export function runAcp() {
  const input = nodeToWebWritable(process.stdout);
  const output = nodeToWebReadable(process.stdin);

  const stream = ndJsonStream(input, output);
  new AgentSideConnection((client) => new ClaudeAcpAgent(client), stream);
}

function commonPrefixLength(a: string, b: string) {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) {
    i++;
  }
  return i;
}

function getMatchingModelUsage(modelUsage: Record<string, ModelUsage>, currentModel: string) {
  let bestKey: string | null = null;
  let bestLen = 0;

  for (const key of Object.keys(modelUsage)) {
    const len = commonPrefixLength(key, currentModel);
    if (len > bestLen) {
      bestLen = len;
      bestKey = key;
    }
  }

  if (bestKey) {
    return modelUsage[bestKey];
  }
}
