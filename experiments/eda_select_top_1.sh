#!/bin/bash

for USER_NUM in 1 2 3 4 5 6 7 8 9
do
	for BOOK_NUM in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
	do
		python main.py --USER_NUM ${USER_NUM} --BOOK_NUM ${BOOK_NUM} --MODEL NCF --VALID random
	done
done