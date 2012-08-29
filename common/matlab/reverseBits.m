function out = reverseBits(inp)

for i = 1:8
    out(i) = inp(9 - i);
end