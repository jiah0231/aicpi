/**
 * Elapsed wall-clock seconds from model generation start to a tool result.
 *
 * The SDK's assistant timestamp is captured before generation, not when the
 * tool starts. This interval includes generation, scheduling/other tools, and
 * any confirmation wait, so it must never be labeled pure tool execution time.
 * Historical messages do not provide a reliable execution-start timestamp.
 */
export function getToolRoundTripSeconds(
  assistantTimestamp: number | undefined,
  resultTimestamp: number | undefined,
): number | undefined {
  if (
    typeof assistantTimestamp !== "number"
    || typeof resultTimestamp !== "number"
    || !Number.isFinite(assistantTimestamp)
    || !Number.isFinite(resultTimestamp)
    || assistantTimestamp < 0
    || resultTimestamp < assistantTimestamp
  ) return undefined;

  const seconds = Math.round((resultTimestamp - assistantTimestamp) / 1000);
  return Number.isFinite(seconds) && seconds > 0 ? seconds : undefined;
}
