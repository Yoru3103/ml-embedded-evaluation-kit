# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Console cancellation uses a native .NET callback, without a PowerShell runspace. #>
if (-not ('Mlek.FvpAudioConsole' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
namespace Mlek {
    public static class FvpAudioConsole {
        private static volatile bool requested;
        public static bool Requested { get { return requested; } }
        private static void Cancel(object sender, ConsoleCancelEventArgs args) {
            args.Cancel = true;
            requested = true;
        }
        public static void Start() {
            requested = false;
            Console.CancelKeyPress += Cancel;
        }
        public static void RequestStop() { requested = true; }
        public static void Stop() { Console.CancelKeyPress -= Cancel; }
    }
}
'@
}

function Test-FvpAudioStopRequested {
    if ([Mlek.FvpAudioConsole]::Requested) { return $true }
    try {
        if (-not [Console]::IsInputRedirected -and [Console]::KeyAvailable) {
            if ([Console]::ReadKey($true).Key -eq [ConsoleKey]::Q) {
                [Mlek.FvpAudioConsole]::RequestStop()
                return $true
            }
        }
    } catch [InvalidOperationException] {
        # Some PowerShell hosts have no console input; Ctrl+C still runs finally.
    }
    return $false
}

function Stop-FvpAudioProcess {
    param([Diagnostics.Process]$Process)
    if ($null -eq $Process) { return }
    try {
        $Process.Refresh()
        if (-not $Process.HasExited) {
            try {
                # Direct .NET cleanup also works while PowerShell cancels its pipeline.
                $Process.Kill()
            } catch [InvalidOperationException] {
                # The child may have exited between HasExited and Kill.
            }
            $null = $Process.WaitForExit(3000)
        }
    } finally {
        $Process.Dispose()
    }
}
