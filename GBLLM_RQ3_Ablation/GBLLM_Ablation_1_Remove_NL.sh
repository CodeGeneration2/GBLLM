Model=DeepSeekV32
dataset=DB
lang=Py
Num_threads=20
Output_Prompt=True

# 519      # deepseek-ai/DeepSeek-V3.2-Exp
Core_Number=519




Prompt=5_Generate_NL_Use_IO__My_Adopted
base_df=Code_Data_Table/${dataset}_${lang}_012_${Model}__Change_Order.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_110
iteration_round=0
Num_gen_codes=1
Num_once_gen=1
Num_rep_gen=1
temperature=0.01






Num_gen_codes=5
Num_once_gen=1
Num_rep_gen=5
temperature=1
Test_IO_type='(Public)'



iteration_round=1

base_df=Code_Data_Table/${dataset}_${lang}_110_${Model}__5_Generate_NL_Use_IO_.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_113_${Model}__Sort_COT_Result_Code_By_Time.csv


Prompt=30_Generate_Code_Ablation_Remove_NL_COT_CFG_Use_IO_Use_Slow_Mid_Fast_Time
base_df=Code_Data_Table/${dataset}_${lang}_139_${Model}__Ablation_Start.csv
Gen_df=Code_Data_Table/${dataset}_${lang}_150
python Large_model_API_generation__latest5.py --prompt_template_name $Prompt --core_number $Core_Number --baseline_df_path $base_df --generated_df_path $Gen_df --iteration_round $iteration_round --num_threads $Num_threads --num_generated_codes $Num_gen_codes --batch_size $Num_once_gen --repeat_times $Num_rep_gen --temperature $temperature


# --is_output_prompt $Output_Prompt