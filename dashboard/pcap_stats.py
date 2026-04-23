#!/usr/bin/env python3
"""Read a PCAP file and print frame-length box-plot stats to stdout.
Output: min q1 median q3 max mean count  (space-separated)
Exits with code 1 if the file cannot be parsed.
"""
import struct, sys, os

def pcap_stats(path):
    try:
        with open(path, 'rb') as f:
            hdr = f.read(24)
            if len(hdr) < 24:
                return None
            magic = struct.unpack('<I', hdr[:4])[0]
            if magic == 0xa1b2c3d4:
                endian = '<'
            elif magic == 0xd4c3b2a1:
                endian = '>'
            else:
                return None
            lengths = []
            while True:
                rec = f.read(16)
                if len(rec) < 16:
                    break
                incl = struct.unpack(endian + 'I', rec[8:12])[0]
                if incl > 65535:
                    break
                data = f.read(incl)
                if len(data) < incl:
                    break
                lengths.append(incl)
        if not lengths:
            return None
        lengths.sort()
        n = len(lengths)
        return (lengths[0], lengths[n // 4], lengths[n // 2],
                lengths[3 * n // 4], lengths[-1],
                round(sum(lengths) / n, 1), n)
    except Exception:
        return None

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)
    result = pcap_stats(sys.argv[1])
    if result is None:
        sys.exit(1)
    print(*result)
