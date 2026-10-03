// Loaded by Windows PowerShell's Add-Type; compatible with its C# compiler.
// No service, scheduled task, browser API, PID-based kill, or stored credentials.
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.NetworkInformation;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace PiWeb
{
    public static class WindowsHost
    {
        private static volatile bool parentClosed;
        private static string directory, runId, mode, phase;
        private static int port, generation;
        private static bool stateWriteFailed;
        private static readonly Encoding Utf8 = new UTF8Encoding(false);

        public static void Run(string root, string node, string launchMode, int launchPort)
        {
            if ((launchMode != "dev" && launchMode != "start") || launchPort < 1 || launchPort > 65535)
                throw new ArgumentException("Invalid launch settings.");
            mode = launchMode;
            port = launchPort;
            directory = Path.Combine(root, ".pi-web-run");
            Directory.CreateDirectory(directory);
            // A crashed owner releases this handle; no stale PID file to trust.
            using (FileStream lease = new FileStream(Path.Combine(directory, "manager.lock"),
                FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read))
            {
                runId = Guid.NewGuid().ToString("N");
                Thread monitor = new Thread(delegate() {
                    try { while (Console.In.Read() != -1) { } }
                    finally { parentClosed = true; }
                });
                monitor.IsBackground = true;
                monitor.Start();
                Environment.SetEnvironmentVariable("PI_WEB_MANAGED_RUN", runId);
                // Matches this launcher's fixed loopback bind; other original
                // variables, including password/allowed hosts/proxies, survive.
                Environment.SetEnvironmentVariable("PI_WEB_HOSTNAME", "127.0.0.1");
                Job job = null;
                Process updater = null;
                bool launch = true;
                DateTime deadline = DateTime.UtcNow;
                phase = "starting";
                try
                {
                    while (!parentClosed)
                    {
                        if (Consume("stop")) break;
                        bool restart = Consume("restart");
                        bool update = ConsumeUpdate();
                        if (updater != null)
                        {
                            // Requests during an update are coalesced, never parallelized.
                            restart = false;
                            if (updater.HasExited)
                            {
                                updater.WaitForExit(); // Drain redirected output before disposing.
                                bool ok = updater.ExitCode == 0;
                                updater.Dispose(); updater = null;
                                ConsumeUpdate();
                                UpdateResult(ok ? "pulled" : "failed");
                                if (ok) { restart = true; Log("Update finished; requesting restart."); }
                                else { phase = "starting"; deadline = DateTime.UtcNow.AddSeconds(120); ClearUpdateLock(); Log("UPDATE FAILED: server was not restarted. See update.log; local files were not reset or cleaned."); }
                            }
                        }
                        else if (update)
                        {
                            restart = false;
                            if (mode != "dev") Log("UPDATE FAILED: source updates require dev mode; no build was run.");
                            else
                            {
                                try
                                {
                                    phase = "updating"; State();
                                    UpdateResult("updating");
                                    ProcessStartInfo info = new ProcessStartInfo(node, Quote(Path.Combine(root, "bin", "web-source-update.mjs")));
                                    info.WorkingDirectory = root;
                                    info.UseShellExecute = false;
                                    info.CreateNoWindow = true;
                                    info.RedirectStandardOutput = true;
                                    info.RedirectStandardError = true;
                                    updater = new Process(); updater.StartInfo = info;
                                    updater.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) { UpdateLog(e.Data); };
                                    updater.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) { UpdateLog(e.Data); };
                                    updater.Start(); updater.BeginOutputReadLine(); updater.BeginErrorReadLine();
                                    Log("UPDATE: pulling origin/gptdot; restart only after success.");
                                }
                                catch (Exception error)
                                {
                                    if (updater != null) { updater.Dispose(); updater = null; }
                                    phase = "starting"; deadline = DateTime.UtcNow.AddSeconds(120);
                                    UpdateResult("failed");
                                    ClearUpdateLock();
                                    Log("UPDATE FAILED: " + error.Message);
                                }
                            }
                        }
                        if (restart)
                        {
                            phase = "restarting";
                            State();
                            Log("Restart requested; stopping this application's job.");
                            // Let the requesting terminal print and return first.
                            Thread.Sleep(1000);
                            if (job != null) { job.Stop(); job.Dispose(); job = null; }
                            launch = true;
                        }
                        if (parentClosed) break;
                        if (launch)
                        {
                            launch = false;
                            generation++;
                            phase = "starting";
                            State();
                            try
                            {
                                // Never take over an existing launcher or kill an
                                // unowned listener. Also refuse cross-mode builds.
                                WaitForFreePort();
                                if (mode == "start" && !File.Exists(Path.Combine(root, ".next", "BUILD_ID")))
                                    throw new IOException("Production build missing; no build was run.");
                                string logPath = Path.Combine(directory, "server-" + runId + "-" + generation + ".log");
                                job = new Job(node, root, mode, port, logPath);
                                deadline = DateTime.UtcNow.AddSeconds(120);
                                Log("Starting " + mode + "; server log: " + Path.GetFileName(logPath));
                            }
                            catch (Exception error)
                            {
                                phase = "failed";
                                Log("FAILED: " + error.Message + " Fix the cause, then request restart.");
                            }
                        }
                        if (job != null)
                        {
                            if (job.HasExited)
                            {
                                job.Stop(); job.Dispose(); job = null;
                                phase = "failed";
                                Log("FAILED: the server exited. See its log, then request restart.");
                            }
                            else
                            {
                                bool owned = job.OwnsListener(port);
                                if (phase == "ready" && !owned)
                                {
                                    phase = "starting";
                                    deadline = DateTime.UtcNow.AddSeconds(120);
                                }
                                if (phase == "starting" && owned && HttpReady() && job.OwnsListener(port))
                                {
                                    phase = "ready";
                                    Log("READY: http://127.0.0.1:" + port + " - refresh your web page.");
                                }
                                else if (phase == "starting" && DateTime.UtcNow > deadline)
                                {
                                    job.Stop(); job.Dispose(); job = null;
                                    phase = "failed";
                                    Log("FAILED: not ready after 120 seconds; stopped this job. Check the server log.");
                                }
                            }
                        }
                        State();
                        Thread.Sleep(500);
                    }
                }
                finally
                {
                    try { if (updater != null) { updater.WaitForExit(); updater.Dispose(); } if (job != null) { try { job.Stop(); } finally { job.Dispose(); } } }
                    finally { phase = "stopped"; State(); Log("STOPPED"); }
                }
            }
        }

        private static void State()
        {
            string temporary = Path.Combine(directory, "state.tmp");
            string target = Path.Combine(directory, "state");
            Exception failure = null;
            // Windows readers/scanners can briefly deny replacement. A missed
            // heartbeat must not unwind Run() and kill a healthy owned job.
            // Keep publication atomic: never delete/truncate the visible state.
            for (int attempt = 0; attempt < 4; attempt++)
            {
                try
                {
                    long now = (long)(DateTime.UtcNow - new DateTime(1970, 1, 1)).TotalMilliseconds;
                    File.WriteAllText(temporary, runId + "\n" + phase + "\n" + now + "\n" + generation + "\n" + port + "\n" + mode + "\nupdate-v1\n", Utf8);
                    if (File.Exists(target)) File.Replace(temporary, target, null);
                    else File.Move(temporary, target);
                    if (stateWriteFailed) Console.Error.WriteLine("STATE RECOVERED: manager status publication resumed.");
                    stateWriteFailed = false;
                    return;
                }
                catch (IOException error) { failure = error; }
                catch (UnauthorizedAccessException error) { failure = error; }
                if (attempt < 3) Thread.Sleep(50 * (attempt + 1));
            }
            // Retry on the next heartbeat. Readers already reject stale state
            // after 20 seconds; update requests still require the current run,
            // generation and in-memory ready phase. Don't log through file I/O
            // here, or a second locked file could make this warning fatal.
            if (!stateWriteFailed) Console.Error.WriteLine("STATE WARNING: could not publish " + target +
                "; the manager will retry without stopping the server. " + failure.Message);
            stateWriteFailed = true;
        }

        private static void UpdateResult(string result)
        {
            File.WriteAllText(Path.Combine(directory, "update-result"), runId + "\n" + generation + "\n" + result, Utf8);
        }

        private static void ClearUpdateLock()
        {
            string file = Path.Combine(directory, "update-" + runId + "-" + generation + ".lock");
            if (File.Exists(file)) File.Delete(file);
        }

        private static readonly object UpdateLogLock = new object();
        private static void UpdateLog(string message)
        {
            if (message == null) return;
            lock (UpdateLogLock) File.AppendAllText(Path.Combine(directory, "update.log"), message + Environment.NewLine, Utf8);
        }

        private static void Log(string message)
        {
            string line = DateTime.UtcNow.ToString("o") + " " + message;
            Console.WriteLine(line);
            File.AppendAllText(Path.Combine(directory, "manager.log"), line + Environment.NewLine, Utf8);
        }

        private static bool ConsumeUpdate()
        {
            // Stale requests can be published after the API read an earlier
            // generation. Discard them; they must never update a later server.
            string expected = Path.Combine(directory, "update-" + runId + "-" + generation + ".request");
            bool current = false;
            foreach (string request in Directory.GetFiles(directory, "update-" + runId + "-*.request"))
            {
                if (String.Equals(request, expected, StringComparison.OrdinalIgnoreCase) && phase == "ready") current = true;
                File.Delete(request);
            }
            return current;
        }

        private static bool Consume(string action)
        {
            string[] requests = Directory.GetFiles(directory, action + "-" + runId + "-*.request");
            foreach (string request in requests) File.Delete(request);
            return requests.Length > 0;
        }

        private static void WaitForFreePort()
        {
            for (int attempt = 0; attempt < 20; attempt++)
            {
                bool occupied = false;
                foreach (IPEndPoint endpoint in IPGlobalProperties.GetIPGlobalProperties().GetActiveTcpListeners())
                    if (endpoint.Port == port) occupied = true;
                if (!occupied) return;
                if (parentClosed) throw new IOException("External launcher closed.");
                Thread.Sleep(250);
            }
            throw new IOException("Port " + port + " is occupied. Stop the old launcher yourself; no existing process was killed.");
        }

        private static bool HttpReady()
        {
            // Check the actual frontend without credentials or model calls.
            // Password-enabled installs must also serve their login page.
            Uri home = new Uri("http://127.0.0.1:" + port + "/");
            return PageReady(home, true);
        }

        private static bool PageReady(Uri uri, bool allowLogin)
        {
            HttpWebRequest request = (HttpWebRequest)WebRequest.Create(uri);
            request.Proxy = null;
            request.AllowAutoRedirect = false;
            request.Timeout = 1500;
            request.ReadWriteTimeout = 1500;
            try
            {
                using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                {
                    if (response.StatusCode == HttpStatusCode.OK) return true;
                    int code = (int)response.StatusCode;
                    Uri login;
                    if (!allowLogin || (code != 302 && code != 303 && code != 307 && code != 308) ||
                        !Uri.TryCreate(uri, response.Headers["Location"], out login) ||
                        login.Scheme != uri.Scheme || login.Host != uri.Host || login.Port != uri.Port ||
                        login.AbsolutePath != "/login" || login.UserInfo.Length != 0) return false;
                    return PageReady(login, false);
                }
            }
            catch (WebException error)
            {
                if (error.Response != null) error.Response.Close();
                return false;
            }
        }

        private sealed class Job : IDisposable
        {
            private IntPtr handle, process;

            internal Job(string node, string root, string launchMode, int launchPort, string logPath)
            {
                IntPtr output = IntPtr.Zero, input = IntPtr.Zero;
                ProcessInfo child = new ProcessInfo();
                try
                {
                    handle = CreateJobObject(IntPtr.Zero, null); // unnamed, not inheritable
                    Check(handle != IntPtr.Zero);
                    ExtendedLimits limits = new ExtendedLimits();
                    limits.Basic.LimitFlags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                    Check(SetInformationJobObject(handle, 9, ref limits, (uint)Marshal.SizeOf(limits)));
                    SecurityAttributes security = new SecurityAttributes();
                    security.Length = Marshal.SizeOf(security);
                    security.InheritHandle = 1;
                    output = CreateFile(logPath, 4, 3, ref security, 4, 0x80, IntPtr.Zero); // append, share read/write
                    Check(output != new IntPtr(-1));
                    input = CreateFile("NUL", 0x80000000, 3, ref security, 3, 0x80, IntPtr.Zero);
                    Check(input != new IntPtr(-1));
                    StartupInfo startup = new StartupInfo();
                    startup.Size = Marshal.SizeOf(startup);
                    startup.Flags = 0x100; // STARTF_USESTDHANDLES
                    startup.Input = input;
                    startup.Output = startup.Error = output;
                    string next = Path.Combine(root, "node_modules", "next", "dist", "bin", "next");
                    StringBuilder command = new StringBuilder(Quote(node) + " " + Quote(next) + " " + launchMode + " -H 127.0.0.1 -p " + launchPort);
                    // Environment is inherited in memory. Suspend before Node can
                    // fork: no listener/worker can escape ownership registration.
                    Check(CreateProcess(node, command, IntPtr.Zero, IntPtr.Zero, true,
                        0x08000004, IntPtr.Zero, root, ref startup, out child)); // NO_WINDOW | SUSPENDED
                    process = child.Process;
                    Check(AssignProcessToJobObject(handle, process));
                    Check(ResumeThread(child.Thread) != 0xffffffff);
                }
                catch
                {
                    // This exact returned handle, never a recycled PID. Covers a
                    // failed assignment while the new process is still suspended.
                    if (process != IntPtr.Zero) TerminateProcess(process, 1);
                    Dispose();
                    throw;
                }
                finally
                {
                    if (child.Thread != IntPtr.Zero) CloseHandle(child.Thread);
                    if (output != IntPtr.Zero && output != new IntPtr(-1)) CloseHandle(output);
                    if (input != IntPtr.Zero && input != new IntPtr(-1)) CloseHandle(input);
                }
            }

            internal bool HasExited { get { return WaitForSingleObject(process, 0) == 0; } }

            internal void Stop()
            {
                Check(TerminateJobObject(handle, 1));
                DateTime deadline = DateTime.UtcNow.AddSeconds(10);
                while (true)
                {
                    Accounting info;
                    Check(QueryInformationJobObject(handle, 1, out info, (uint)Marshal.SizeOf(typeof(Accounting)), IntPtr.Zero));
                    if (info.ActiveProcesses == 0) return;
                    if (DateTime.UtcNow > deadline) throw new IOException("This job is still stopping; refusing to launch another server.");
                    Thread.Sleep(100);
                }
            }

            internal bool OwnsListener(int listenerPort)
            {
                List<uint> owners = ListenerOwners(listenerPort);
                if (owners.Count == 0) return false;
                foreach (uint owner in owners)
                {
                    IntPtr candidate = OpenProcess(0x1000, false, owner);
                    if (candidate == IntPtr.Zero) return false;
                    try
                    {
                        bool belongs;
                        if (!IsProcessInJob(candidate, handle, out belongs) || !belongs) return false;
                    }
                    finally { CloseHandle(candidate); }
                }
                return true;
            }

            public void Dispose()
            {
                if (handle != IntPtr.Zero) { CloseHandle(handle); handle = IntPtr.Zero; }
                if (process != IntPtr.Zero) { CloseHandle(process); process = IntPtr.Zero; }
            }
        }

        private static string Quote(string value)
        {
            // These are existing Windows file paths (node.exe and next), never
            // shell text, and neither ends in a directory separator.
            if (value.IndexOf('"') >= 0 || value.EndsWith("\\")) throw new ArgumentException("Invalid executable path.");
            return "\"" + value + "\"";
        }

        private static void Check(bool ok)
        {
            if (!ok) throw new Win32Exception(Marshal.GetLastWin32Error());
        }

        private static List<uint> ListenerOwners(int listenerPort)
        {
            int size = 0;
            uint result = GetExtendedTcpTable(IntPtr.Zero, ref size, false, 2, 3, 0); // IPv4, OWNER_PID_LISTENER
            if (result != 122) throw new Win32Exception((int)result);
            for (int attempt = 0; attempt < 3; attempt++)
            {
                IntPtr table = Marshal.AllocHGlobal(size);
                try
                {
                    result = GetExtendedTcpTable(table, ref size, false, 2, 3, 0);
                    if (result == 122) continue;
                    if (result != 0) throw new Win32Exception((int)result);
                    List<uint> owners = new List<uint>();
                    int count = Marshal.ReadInt32(table);
                    for (int index = 0; index < count; index++)
                    {
                        int offset = 4 + index * 24; // MIB_TCPROW_OWNER_PID
                        int rowPort = (Marshal.ReadByte(table, offset + 8) << 8) | Marshal.ReadByte(table, offset + 9);
                        if (rowPort == listenerPort) owners.Add(unchecked((uint)Marshal.ReadInt32(table, offset + 20)));
                    }
                    return owners;
                }
                finally { Marshal.FreeHGlobal(table); }
            }
            throw new IOException("Listener ownership changed while checking; refusing to report ready.");
        }

        [StructLayout(LayoutKind.Sequential)] private struct BasicLimits
        {
            public long ProcessTime, JobTime;
            public uint LimitFlags;
            public UIntPtr MinWorkingSet, MaxWorkingSet;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint Priority, SchedulingClass;
        }
        [StructLayout(LayoutKind.Sequential)] private struct ExtendedLimits
        {
            public BasicLimits Basic;
            public ulong ReadOperations, WriteOperations, OtherOperations, ReadBytes, WriteBytes, OtherBytes;
            public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
        }
        [StructLayout(LayoutKind.Sequential)] private struct Accounting
        {
            public long UserTime, KernelTime, PeriodUserTime, PeriodKernelTime;
            public uint PageFaults, TotalProcesses, ActiveProcesses, TerminatedProcesses;
        }
        [StructLayout(LayoutKind.Sequential)] private struct SecurityAttributes
        { public int Length; public IntPtr Descriptor; public int InheritHandle; }
        [StructLayout(LayoutKind.Sequential)] private struct ProcessInfo
        { public IntPtr Process, Thread; public uint ProcessId, ThreadId; }
        [StructLayout(LayoutKind.Sequential)] private struct StartupInfo
        {
            public int Size;
            public IntPtr Reserved, Desktop, Title;
            public uint X, Y, Width, Height, XChars, YChars, Fill, Flags;
            public ushort ShowWindow, ReservedSize;
            public IntPtr ReservedData, Input, Output, Error;
        }
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateJobObject(IntPtr security, string name);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref ExtendedLimits info, uint size);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool QueryInformationJobObject(IntPtr job, int infoClass, out Accounting info, uint size, IntPtr returned);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool TerminateJobObject(IntPtr job, uint exitCode);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool result);
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr OpenProcess(uint access, bool inherit, uint id);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool CreateProcess(string application, StringBuilder command, IntPtr processSecurity,
            IntPtr threadSecurity, bool inherit, uint flags, IntPtr environment, string cwd, ref StartupInfo startup, out ProcessInfo process);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateFile(string path, uint access, uint share, ref SecurityAttributes security,
            uint disposition, uint flags, IntPtr template);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern uint ResumeThread(IntPtr thread);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern bool TerminateProcess(IntPtr process, uint exitCode);
        [DllImport("kernel32.dll", SetLastError = true)] private static extern uint WaitForSingleObject(IntPtr handle, uint timeout);
        [DllImport("kernel32.dll")] private static extern bool CloseHandle(IntPtr handle);
        [DllImport("iphlpapi.dll")] private static extern uint GetExtendedTcpTable(IntPtr table, ref int size,
            bool ordered, uint addressFamily, int tableClass, uint reserved);
    }
}
