
AUXFILE=.aux_file
max_test=$1

printf -v line '\n%*s' "$length"
line=${line// /-}

if [[ -z $max_test ]]
then
	max_test=1000
fi
array=( $@ )
len=${#array[@]}
args=${array[@]:1:$len}
c=0
while [[ True ]]
do 
	c=$((c+1))
	echo -e $line
	echo "Test ${c}/${max_test} :"
	echo $line
	echo
	for ver in 27 33 34 35
	do
		tox -e py$ver $args 2> $AUXFILE
		ret_val=$?

		#test $ret_val -ne 0 && cat $AUXFILE

		# Break if one test fail or if we reached the $max_test value
		test $ret_val -ne 0 && exit 1
		test $c -ge $max_test && rm $AUXFILE && exit 1
    done
done
