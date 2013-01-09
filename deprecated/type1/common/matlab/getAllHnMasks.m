function hnMasks = getAllHnMasks()

%1-based indexing for cNums and eTypes
%There are only 18 valid hanging types depending on the childnumber.
%Values in the same order as in the C++ code (oda.h), if node i (0-based
%indexing) is hanging then (1 << i) is set to 1.

hnMasks(1, :) = [0, 4, 2, 6, 16, 20, 18, 22, 14, 30, 84, 86, 94, 50, 54, 62, 118, 126];
hnMasks(2, :) = [0, 1, 8, 9, 32, 33, 40, 41, 13, 45, 49, 57, 61, 168, 169, 173, 185, 189];
hnMasks(3, :) = [0, 8, 1, 9, 64, 72, 65, 73, 11, 75, 200, 201, 203, 81, 89, 91, 217, 219];
hnMasks(4, :) = [0, 2, 4, 6, 128, 130, 132, 134, 7, 135, 162, 166, 167, 196, 198, 199, 230, 231];
hnMasks(5, :) = [0, 1, 32, 33, 64, 65, 96, 97, 35, 99, 69, 101, 103, 224, 225, 227, 229, 231];
hnMasks(6, :) = [0, 2, 128, 130, 16, 18, 144, 146, 138, 154, 19, 147, 155, 208, 210, 218, 211, 219];
hnMasks(7, :) = [0, 4, 16, 20, 128, 132, 144, 148, 21, 149, 140, 156, 157, 176, 180, 181, 188, 189];
hnMasks(8, :) = [0, 8, 64, 72, 32, 40, 96, 104, 76, 108, 42, 106, 110, 112, 120, 124, 122, 126];


