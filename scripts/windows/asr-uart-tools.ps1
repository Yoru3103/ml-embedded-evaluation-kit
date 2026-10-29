# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# WAV and CRC helpers shared by the sender and its host-side tests. #>
if (-not ('Mlek.AsrAudio' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Text;
namespace Mlek {
    public sealed class AsrAudio {
        public byte[] Pcm;
        public uint Samples;
        public uint Crc;
        public static uint Crc32(byte[] data) {
            uint crc = 0xffffffff;
            foreach (byte value in data) {
                crc ^= value;
                for (int i = 0; i < 8; ++i)
                    crc = (crc >> 1) ^ ((crc & 1) != 0 ? 0xedb88320U : 0U);
            }
            return crc ^ 0xffffffff;
        }
        public static AsrAudio Load(string path) {
            using (var reader = new BinaryReader(File.OpenRead(path))) {
                if (reader.BaseStream.Length < 12 ||
                    Encoding.ASCII.GetString(reader.ReadBytes(4)) != "RIFF")
                    throw new InvalidDataException("Requires a RIFF PCM WAV file.");
                long end = (long)reader.ReadUInt32() + 8;
                if (end > reader.BaseStream.Length || end < 12 ||
                    Encoding.ASCII.GetString(reader.ReadBytes(4)) != "WAVE")
                    throw new InvalidDataException("Invalid WAV container.");
                bool format = false;
                byte[] pcm = null;
                while (reader.BaseStream.Position + 8 <= end) {
                    string id = Encoding.ASCII.GetString(reader.ReadBytes(4));
                    uint size = reader.ReadUInt32();
                    long start = reader.BaseStream.Position;
                    long next = start + size + (size & 1);
                    if (next > end) throw new InvalidDataException("Truncated WAV chunk.");
                    if (id == "fmt ") {
                        if (format || size < 16) throw new InvalidDataException("Invalid fmt chunk.");
                        ushort encoding = reader.ReadUInt16(), channels = reader.ReadUInt16();
                        uint rate = reader.ReadUInt32(), byteRate = reader.ReadUInt32();
                        ushort align = reader.ReadUInt16(), bits = reader.ReadUInt16();
                        if (encoding != 1 || channels != 1 || rate != 16000 ||
                            bits != 16 || align != 2 || byteRate != 32000)
                            throw new InvalidDataException("Requires 16000 Hz, mono, signed PCM16 (format 1).");
                        format = true;
                    } else if (id == "data") {
                        if (pcm != null || size < 1024 || size > 960000 || (size & 1) != 0)
                            throw new InvalidDataException("PCM length must be 512..480000 samples (max 30 seconds).");
                        pcm = reader.ReadBytes((int)size);
                        if (pcm.Length != size) throw new InvalidDataException("Truncated PCM.");
                    }
                    reader.BaseStream.Position = next;
                }
                if (!format || pcm == null) throw new InvalidDataException("Missing fmt or data chunk.");
                return new AsrAudio { Pcm = pcm, Samples = (uint)pcm.Length / 2, Crc = Crc32(pcm) };
            }
        }
        public byte[] Header(uint requestId) {
            using (var stream = new MemoryStream()) {
                using (var writer = new BinaryWriter(stream)) {
                    writer.Write(Encoding.ASCII.GetBytes("ASR1"));
                    writer.Write(requestId); writer.Write(Samples); writer.Write(Crc);
                    writer.Flush();
                    return stream.ToArray();
                }
            }
        }
    }
}
'@
}
