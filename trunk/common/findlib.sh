#!/bin/bash
# to find a library that contains a given symbol
#Use:
#./findlib.sh sym libs
sym="$1"
shift
for name do
   echo $name
   nm -C $name | grep $sym
done

