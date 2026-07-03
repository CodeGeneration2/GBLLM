# -------------------------- Configuration Variables --------------------------

# [Model] Specifies the Large Language Model (LLM) to be used.
# Options: 
#   - DeepSeekV32: deepseek-ai/DeepSeek-V3.2-Exp (Core Index: 519)
#   - CodeLlama13B: CodeLlama-13b-Instruct-hf (Core Index: 101)
#   - CodeLlama34B: CodeLlama-34b-Instruct-hf (Core Index: 110)
#   - GPT3: gpt-3.5-turbo-0125 (Core Index: 319)
#   - Gemini: Gemini-2.5-flash (Core Index: 239)
Model=DeepSeekV32

# [dataset] Target dataset for evaluation.
# Options: PIE, PPIE
dataset=PIE

# [lang] Programming language of the source code.
# Options: Py (Python), Cpp (C++)
lang=Py

# [Num_threads] Number of parallel threads.
# Increase this value to speed up processing; monitor API rate limits accordingly.
Num_threads=1

# [Output_Prompt] Debugging flag.
# Set to 'True' to print the generated prompts to the console for troubleshooting.
Output_Prompt=False

# [Core_Number] Unique identifier mapped to the selected Model.
# 101: CodeLlama13B | 110: CodeLlama34B | 239: Gemini | 319: GPT3 | 519: DeepSeekV32
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