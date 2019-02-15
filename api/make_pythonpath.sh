main_path='/home/ubuntu/projects/'
parts=($main_path'cat')

py_path=$(printf ":%s" "${parts[@]}")
py_path=${py_path:1}

echo "This is the new PYTHONPATH:"
export PYTHONPATH=$py_path
echo "\t" $PYTHONPATH
