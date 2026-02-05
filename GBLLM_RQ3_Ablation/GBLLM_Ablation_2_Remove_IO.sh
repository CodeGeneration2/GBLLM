Model=DeepSeekV32
dataset=DB
lang=Py
Num_threads=20
Output_Prompt=True

# 519      # deepseek-ai/DeepSeek-V3.2-Exp
Core_Number=519



Prompt=32_Generate_NL_Ablation_Remove_IO_Long_NL
base_df=Code_Data_Table/${dataset}_${lang}_150_${Model}__30_Generate_Code_Ablation_Remove_NL_COT_CFG_Use_IO_Use_Slow_Mid_Fast_Time_.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_152
iteration_round=0
Num_gen_codes=1
Num_once_gen=1
Num_rep_gen=1
temperature=0.01
# --is_output_prompt $Output_Prompt
python Large_model_API_generation__latest5.py --prompt_template_name $Prompt --core_number $Core_Number --baseline_df_path $base_df --generated_df_path $Gen_df --iteration_round $iteration_round --num_threads $Num_threads --num_generated_codes $Num_gen_codes --batch_size $Num_once_gen --repeat_times $Num_rep_gen --temperature $temperature






Num_gen_codes=5
Num_once_gen=1
Num_rep_gen=5
temperature=1
Test_IO_type='(Public)'



iteration_round=1

base_df=Code_Data_Table/${dataset}_${lang}_110_${Model}__5_Generate_NL_Use_IO_.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_113_${Model}__Sort_COT_Result_Code_By_Time.csv


Prompt=33_Generate_Code_Ablation_Remove_IO_COT_CFG_Use_NL_Use_Slow_Mid_Fast_Time
base_df=Code_Data_Table/${dataset}_${lang}_152_${Model}__32_Generate_NL_Ablation_Remove_IO_Long_NL_.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_155

NL=Ablation_Remove_IO_Code_Function_Description_G1

python Large_model_API_generation__latest5.py  --core_number $Core_Number --nl_column $NL --prompt_template_name $Prompt --baseline_df_path $base_df --generated_df_path $Gen_df --iteration_round $iteration_round --num_threads $Num_threads --num_generated_codes $Num_gen_codes --batch_size $Num_once_gen --repeat_times $Num_rep_gen --temperature $temperature