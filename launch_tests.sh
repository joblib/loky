
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

	if [[ $c -ge $max_test || $ret_val -ne 0 ]]
	then 
		sed -n -e '/[Pp]ool /!p' -i $AUXFILE 
		cat $AUXFILE
    	rm $AUXFILE
		break
	fi
    rm $AUXFILE
done
