
AUXFILE=.aux_file
max_test=$1
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
	echo "Test ${c}/${max_test} :"
	tox $args 2>$AUXFILE
	ret_val=$?

	test $ret_val -ne 0 && cat $AUXFILE
	rm $AUXFILE

	# Break if one test fail or if we reached the $max_test value
	test $c -ge $max_test -o $ret_val -ne 0 && break
done
