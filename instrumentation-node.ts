import { startGroundingLearningWorker, stopGroundingLearningWorker } from "@/lib/grounding-learning-worker";
import { configureHttpDispatcher } from "@/lib/http-dispatcher";
import { closeAllAgentEventStreams } from "@/lib/agent-event-stream";

export function registerNodeInstrumentation(): void {
  configureHttpDispatcher();
  startGroundingLearningWorker();

  // In production Next 16 answers SIGINT/SIGTERM with server.close() and waits
  // for every connection to end, without a timeout. SSE streams only end when
  // the client disconnects, so close them here or the process never exits.
  const shutdownStreams = () => { stopGroundingLearningWorker(); closeAllAgentEventStreams(); };
  process.on("SIGINT", shutdownStreams);
  process.on("SIGTERM", shutdownStreams);
}
