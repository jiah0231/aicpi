import { NextResponse } from "next/server";
import { managedUpdateRoot, readUpdateManager, requestSourceUpdate } from "@/lib/source-update";

export const dynamic = "force-dynamic";
export async function GET() {
  try {
    const { root, run } = await managedUpdateRoot();
    return NextResponse.json(await readUpdateManager(root, run));
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status: 409 });
  }
}
export async function POST(request: Request) {
  try {
    const body = await request.json();
    if (body?.confirmInterruption !== true || Object.keys(body).some((key) => key !== "confirmInterruption")) {
      return NextResponse.json({ error: "Confirm that restart will interrupt all active tasks, reviews and embedded terminals. No other arguments are accepted." }, { status: 400 });
    }
    const { root, run } = await managedUpdateRoot();
    return NextResponse.json(await requestSourceUpdate(root, run), { status: 202 });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status: 409 });
  }
}
