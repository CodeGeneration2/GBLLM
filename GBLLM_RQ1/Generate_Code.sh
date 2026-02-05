Model=DeepSeekV32
dataset=PIE
lang=Py
Num_threads=1
Output_Prompt=True

# 101    # CodeLlama-13b-Instruct-hf
# 110    # CodeLlama-34b-Instruct-hf
# 239    # Gemini-2.5-flash
# 319    # gpt-3.5-turbo-0125
# 519      # deepseek-ai/DeepSeek-V3.2-Exp
Core_Number=519



Prompt=5_Generate_NL_Use_IO__My_Adopted
base_df=Code_Data_Table/${dataset}_${lang}_010_${Model}__Change_Order.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_110
iteration_round=0
Num_gen_codes=1
Num_once_gen=1
Num_rep_gen=1
temperature=0.01
python Large_model_API_generation.py --is_output_prompt $Output_Prompt --core_number $Core_Number --prompt_template_name $Prompt --baseline_df_path $base_df --generated_df_path $Gen_df --iteration_round $iteration_round --num_threads $Num_threads --num_generated_codes $Num_gen_codes --batch_size $Num_once_gen --repeat_times $Num_rep_gen --temperature $temperature



# --is_output_prompt $Output_Prompt



Num_gen_codes=5
Num_once_gen=1
Num_rep_gen=5
temperature=1
Test_IO_type='(Public)'



iteration_round=1

base_df=Code_Data_Table/${dataset}_${lang}_110_${Model}__5_Generate_NL_Use_IO_.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_113_${Model}__Sort_COT_Result_Code_By_Time.csv
python Sort_COT_result_codes_by_time.py --dataset_path $base_df --save_set_path $Gen_df --prepare_round_number $iteration_round


Prompt=14_Generate_Code_COT_CFG_Use_NL_Use_IO_Use_Slow_Mid_Fast_Time__My_Adopted
base_df=Code_Data_Table/${dataset}_${lang}_113_${Model}__Sort_COT_Result_Code_By_Time.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_115
python Large_model_API_generation.py  --prompt_template_name $Prompt --core_number $Core_Number --baseline_df_path $base_df --generated_df_path $Gen_df --iteration_round $iteration_round --num_threads $Num_threads --num_generated_codes $Num_gen_codes --batch_size $Num_once_gen --repeat_times $Num_rep_gen --temperature $temperature


base_df=Code_Data_Table/${dataset}_${lang}_115_${Model}__14_Generate_Code_COT_CFG_Use_NL_Use_IO_Use_Slow_Mid_Fast_Time_.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_120_${Model}__Evaluate_Code_Execution_Time.csv
python PIE__Evaluate_code_execution_time.py --test_io_type $Test_IO_type --iteration_round $iteration_round --dataset_path $base_df --save_set_path $Gen_df







iteration_round=2

base_df=Code_Data_Table/${dataset}_${lang}_120_${Model}__Evaluate_Code_Execution_Time.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_123_${Model}__Sort_COT_Result_Code_By_Time.csv
python Sort_COT_result_codes_by_time.py --dataset_path $base_df --save_set_path $Gen_df --prepare_round_number $iteration_round


Prompt=14_Generate_Code_COT_CFG_Use_NL_Use_IO_Use_Slow_Mid_Fast_Time__My_Adopted
base_df=Code_Data_Table/${dataset}_${lang}_123_${Model}__Sort_COT_Result_Code_By_Time.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_125
python Large_model_API_generation.py  --prompt_template_name $Prompt --core_number $Core_Number --baseline_df_path $base_df --generated_df_path $Gen_df --iteration_round $iteration_round --num_threads $Num_threads --num_generated_codes $Num_gen_codes --batch_size $Num_once_gen --repeat_times $Num_rep_gen --temperature $temperature



base_df=Code_Data_Table/${dataset}_${lang}_125_${Model}__14_Generate_Code_COT_CFG_Use_NL_Use_IO_Use_Slow_Mid_Fast_Time_.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_130_${Model}__Evaluate_Code_Execution_Time.csv

python PIE__Evaluate_code_execution_time.py --test_io_type $Test_IO_type --iteration_round $iteration_round --dataset_path $base_df --save_set_path $Gen_df








iteration_round=3

base_df=Code_Data_Table/${dataset}_${lang}_130_${Model}__Evaluate_Code_Execution_Time.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_133_${Model}__Sort_COT_Result_Code_By_Time.csv
python Sort_COT_result_codes_by_time.py --dataset_path $base_df --save_set_path $Gen_df --prepare_round_number $iteration_round


Prompt=14_Generate_Code_COT_CFG_Use_NL_Use_IO_Use_Slow_Mid_Fast_Time__My_Adopted
base_df=Code_Data_Table/${dataset}_${lang}_133_${Model}__Sort_COT_Result_Code_By_Time.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_135
python Large_model_API_generation.py  --prompt_template_name $Prompt --core_number $Core_Number --baseline_df_path $base_df --generated_df_path $Gen_df --iteration_round $iteration_round --num_threads $Num_threads --num_generated_codes $Num_gen_codes --batch_size $Num_once_gen --repeat_times $Num_rep_gen --temperature $temperature



base_df=Code_Data_Table/${dataset}_${lang}_135_${Model}__14_Generate_Code_COT_CFG_Use_NL_Use_IO_Use_Slow_Mid_Fast_Time_.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_140_${Model}__Evaluate_Code_Execution_Time.csv

python PIE__Evaluate_code_execution_time.py --test_io_type $Test_IO_type --iteration_round $iteration_round --dataset_path $base_df --save_set_path $Gen_df







Test_IO_type='(Private)'

iteration_round=3
base_df=Code_Data_Table/${dataset}_${lang}_140_${Model}__Evaluate_Code_Execution_Time.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_142_${Model}__Evaluate_Code_Private_Execution_Time.csv
python PIE__Evaluate_code_execution_time.py --test_io_type $Test_IO_type --iteration_round $iteration_round --dataset_path $base_df --save_set_path $Gen_df

iteration_round=2
base_df=Code_Data_Table/${dataset}_${lang}_142_${Model}__Evaluate_Code_Private_Execution_Time.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_144_${Model}__Evaluate_Code_Private_Execution_Time.csv
python PIE__Evaluate_code_execution_time.py --test_io_type $Test_IO_type --iteration_round $iteration_round --dataset_path $base_df --save_set_path $Gen_df


iteration_round=1
base_df=Code_Data_Table/${dataset}_${lang}_144_${Model}__Evaluate_Code_Private_Execution_Time.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_146_${Model}__Evaluate_Code_Private_Execution_Time.csv
python PIE__Evaluate_code_execution_time.py --test_io_type $Test_IO_type --iteration_round $iteration_round --dataset_path $base_df --save_set_path $Gen_df