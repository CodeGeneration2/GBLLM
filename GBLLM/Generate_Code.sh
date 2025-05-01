model_name=CodeGeneration2/CodeLlama-34b-Instruct-hf
api_key=''

base_data_path=../processed_data/PIE__test.csv
output_path=../NL
role_key=english_generate_nl
prompt_key=english_generate_nl_io
python Large_model_API_generation.py  --api_key $api_key   --output_path $output_path --model_name $model_name --base_data_path $base_data_path --generation_number 5 --role_key $role_key --prompt_key $prompt_key

base_data_path=../NL/PIE__test__CodeLlama.csv
output_path=../output
role_key=english_triple_quotes
prompt_key=english_triple_quotes_cfg_io_use_nl
python Large_model_API_generation.py  --api_key $api_key   --output_path $output_path --model_name $model_name --base_data_path $base_data_path --generation_number 5 --role_key $role_key --prompt_key $prompt_key




model_name=gemini-1.5-pro
api_key=''

base_data_path=../processed_data/PIE__test.csv
output_path=../NL
role_key=english_generate_nl
prompt_key=english_generate_nl_io
python Large_model_API_generation.py  --api_key $api_key   --output_path $output_path --model_name $model_name --base_data_path $base_data_path --generation_number 5 --role_key $role_key --prompt_key $prompt_key

base_data_path=../NL/PIE__test__Gemini.csv
output_path=../output
role_key=english_triple_quotes
prompt_key=english_triple_quotes_cfg_io_use_nl
python Large_model_API_generation.py  --api_key $api_key   --output_path $output_path --model_name $model_name --base_data_path $base_data_path --generation_number 5 --role_key $role_key --prompt_key $prompt_key





model_name=gpt-3.5-turbo
api_key=''

base_data_path=../processed_data/PIE__test.csv
output_path=../NL
role_key=english_generate_nl
prompt_key=english_generate_nl_io
python Large_model_API_generation.py  --api_key $api_key   --output_path $output_path --model_name $model_name --base_data_path $base_data_path --generation_number 5 --role_key $role_key --prompt_key $prompt_key

base_data_path=../NL/PIE__test__GPT3.csv
output_path=../output
role_key=english_triple_quotes
prompt_key=english_triple_quotes_cfg_io_use_nl
python Large_model_API_generation.py  --api_key $api_key   --output_path $output_path --model_name $model_name --base_data_path $base_data_path --generation_number 5 --role_key $role_key --prompt_key $prompt_key




model_name=gpt-4-1106-preview
api_key=''

base_data_path=../processed_data/PIE__test__GPT4.csv
output_path=../NL
role_key=english_generate_nl
prompt_key=english_generate_nl_io
python Large_model_API_generation.py  --api_key $api_key   --output_path $output_path --model_name $model_name --base_data_path $base_data_path --generation_number 5 --role_key $role_key --prompt_key $prompt_key

base_data_path=../NL/PIE__test__GPT4.csv
output_path=../output
role_key=english_triple_quotes
prompt_key=english_triple_quotes_cfg_io_use_nl
python Large_model_API_generation.py  --api_key $api_key   --output_path $output_path --model_name $model_name --base_data_path $base_data_path --generation_number 5 --role_key $role_key --prompt_key $prompt_key

