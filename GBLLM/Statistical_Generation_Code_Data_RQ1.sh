dataset_path=../output/PIE__test__CodeLlama.csv
column_prefix='Use_NL_T0_01_IO'
mode=top1
python Statistical_Generation_Code_Data.py  --dataset_path $dataset_path --column_prefix $column_prefix --mode $mode

dataset_path=../output/PIE__test__CodeLlama.csv
column_prefix='Use_NL_T0_01_IO'
mode=top3
python Statistical_Generation_Code_Data.py  --dataset_path $dataset_path --column_prefix $column_prefix --mode $mode


dataset_path=../output/PIE__test__CodeLlama.csv
column_prefix='Use_NL_T0_01_IO'
mode=top5
python Statistical_Generation_Code_Data.py  --dataset_path $dataset_path --column_prefix $column_prefix --mode $mode




dataset_path=../output/PIE__test__Gemini.csv
column_prefix='Use_NL_T0_01_IO'
mode=top1
python Statistical_Generation_Code_Data.py  --dataset_path $dataset_path --column_prefix $column_prefix --mode $mode

dataset_path=../output/PIE__test__Gemini.csv
column_prefix='Use_NL_T0_01_IO'
mode=top3
python Statistical_Generation_Code_Data.py  --dataset_path $dataset_path --column_prefix $column_prefix --mode $mode

dataset_path=../output/PIE__test__Gemini.csv
column_prefix='Use_NL_T0_01_IO'
mode=top5
python Statistical_Generation_Code_Data.py  --dataset_path $dataset_path --column_prefix $column_prefix --mode $mode



dataset_path=../output/PIE__test__GPT3.csv
column_prefix='Use_NL_T0_01_IO'
mode=top1
python Statistical_Generation_Code_Data.py  --dataset_path $dataset_path --column_prefix $column_prefix --mode $mode

dataset_path=../output/PIE__test__GPT3.csv
column_prefix='Use_NL_T0_01_IO'
mode=top3
python Statistical_Generation_Code_Data.py  --dataset_path $dataset_path --column_prefix $column_prefix --mode $mode

dataset_path=../output/PIE__test__GPT3.csv
column_prefix='Use_NL_T0_01_IO'
mode=top5
python Statistical_Generation_Code_Data.py  --dataset_path $dataset_path --column_prefix $column_prefix --mode $mode




dataset_path=../output/PIE__test__GPT4.csv
column_prefix='Use_NL_T0_01_IO'
mode=top1
python Statistical_Generation_Code_Data.py  --dataset_path $dataset_path --column_prefix $column_prefix --mode $mode

dataset_path=../output/PIE__test__GPT4.csv
column_prefix='Use_NL_T0_01_IO'
mode=top3
python Statistical_Generation_Code_Data.py  --dataset_path $dataset_path --column_prefix $column_prefix --mode $mode

dataset_path=../output/PIE__test__GPT4.csv
column_prefix='Use_NL_T0_01_IO'
mode=top5
python Statistical_Generation_Code_Data.py  --dataset_path $dataset_path --column_prefix $column_prefix --mode $mode
