# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Raw TCP adapter for the FVP UART; preserves partial lines across read timeouts. #>
if (-not ('Mlek.AsrTcpTransport' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Net.Sockets;
using System.Text;

namespace Mlek {
    public sealed class AsrTcpTransport : IDisposable {
        private readonly string host;
        private readonly int port;
        private readonly int connectTimeout;
        private TcpClient client;
        private NetworkStream stream;
        private readonly StringBuilder partialLine = new StringBuilder();
        public bool IsOpen { get { return stream != null; } }

        public AsrTcpTransport(string host, int port, int connectTimeout) {
            this.host = host;
            this.port = port;
            this.connectTimeout = connectTimeout;
        }

        public void Open() {
            client = new TcpClient();
            try {
                var pending = client.BeginConnect(host, port, null, null);
                using (pending.AsyncWaitHandle) {
                    if (!pending.AsyncWaitHandle.WaitOne(connectTimeout)) {
                        throw new TimeoutException("FVP TCP connection timed out.");
                    }
                    client.EndConnect(pending);
                }
                client.NoDelay = true;
                stream = client.GetStream();
                stream.ReadTimeout = 500;
                stream.WriteTimeout = 5000;
            } catch {
                Close();
                throw;
            }
        }

        public void DiscardInBuffer() {
            partialLine.Clear();
            var buffer = new byte[4096];
            while (stream.DataAvailable) {
                if (stream.Read(buffer, 0, buffer.Length) == 0) {
                    throw new EndOfStreamException("FVP closed the connection.");
                }
            }
        }

        public void Write(string text) {
            byte[] bytes = Encoding.UTF8.GetBytes(text);
            Write(bytes, 0, bytes.Length);
        }

        public void Write(byte[] bytes, int offset, int count) {
            stream.Write(bytes, offset, count);
        }

        public string ReadLine() {
            while (true) {
                int value;
                try { value = stream.ReadByte(); }
                catch (IOException error) {
                    var socket = error.InnerException as SocketException;
                    if (socket != null && (socket.SocketErrorCode == SocketError.TimedOut ||
                                           socket.SocketErrorCode == SocketError.WouldBlock)) {
                        throw new TimeoutException("FVP UART read timed out.", error);
                    }
                    throw;
                }
                if (value < 0) { throw new EndOfStreamException("FVP closed the connection."); }
                if (value == '\n') {
                    string line = partialLine.ToString();
                    partialLine.Clear();
                    return line;
                }
                if (partialLine.Length >= 16384) {
                    throw new InvalidDataException("FVP UART response line is too long.");
                }
                partialLine.Append((char)value);
            }
        }

        public void Close() {
            if (stream != null) { stream.Dispose(); stream = null; }
            if (client != null) { client.Close(); client = null; }
            partialLine.Clear();
        }
        public void Dispose() { Close(); }
    }
}
'@
}
