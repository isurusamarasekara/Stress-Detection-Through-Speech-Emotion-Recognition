# Stress Detection Through Speech Analysis and Variations in Vocal Indices

## Technicals
### Languages used   
- Python  
### IDEs  
- Pycharm  
### Hardware  
- Raspberry Pi - Model 4B - 4 GiB  
- USB Microphone - 48 kHz of maximum sampling frequency capability  
  
  
## Directory Structure of the code repo
1. backbone - contains the research code representing everything from data analysis to model training
2. backbone_independent - Windows based realtime speech stress prediction and upload-file speech stress prediction independent from the training packages at 'backbone'.
3. speech_analysis_raspi - The raspberry pi optimized speech stress analysis component This is a complete working code where just copying this folder and running one of the prediction scripts in a virtual environment, where the required python packages are installed, is enough to get this up ad running. Find the 'requirements.txt' file inside this folder for raspberry pi's production python environment relevant for speech stress prediction.

### Important
The trained cleaned audio file directory, model logs directory, dataframes, features, modelconfigs, models and training metrics folders are not pushed since they contain no scriptable code but large data files.
For a better understanding of the expected file structure, read below.

(The below structure is properly visible when opened with notepad or a similar client)
(d - directory)

.  
├── backbone(d)  
│   ├── base_store(d)  
│   │   ├── clean_audio(d)  
│   │   │   ├── cremad_f(d)  
│   │   │   ├── cremad_m(d)  
│   │   │   ├── emodb_f(d)  
│   │   │   ├── emodb_m(d)  
│   │   │   ├── ravdess_f(d)  
│   │   │   ├── ravdess_m(d)  
│   │   │   ├── shemo_f(d)  
│   │   │   └──  shemo_m(d)  
│   │   │  
│   │   ├── logs(d)  
│   │   │  
│   │   ├── saved_dataframes(d)  
│   │   │  
│   │   ├── saved_features(d)  
│   │   │   ├── cremad(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   ├── emodb(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   ├── ravdess(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   └── shemo(d)  
│   │   │       ├── female(d)   
│   │   │       └── male(d)  
│   │   │  
│   │   ├── saved_modelconfigs(d)  
│   │   │   ├── cremad(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   ├── emodb(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   ├── ravdess(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   └── shemo(d)  
│   │   │       ├── female(d)   
│   │   │       └── male(d)  
│   │   │  
│   │   ├── saved_models(d)  
│   │   │   ├── cremad(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   ├── emodb(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   ├── ravdess(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   └── shemo(d)  
│   │   │       ├── female(d)   
│   │   │       └── male(d)  
│   │   │  
│   │   └── saved_training_metrics_logs(d)  
│   │       ├── cremad(d)  
│   │       │   ├── female(d)   
│   │       │   └── male(d)  
│   │       ├── emodb(d)  
│   │       │   ├── female(d)   
│   │       │   └── male(d)  
│   │       ├── ravdess(d)  
│   │       │   ├── female(d)   
│   │       │   └── male(d)  
│   │       └── shemo(d)  
│   │           ├── female(d)   
│   │           └── male(d)  
│   │  
│   ├── mains-dataset_wise_structural(d)  
│   │   ├── main_cremad_female.py  
│   │   ├── main_cremad_male.py  
│   │   ├── main_emodb_female.py  
│   │   ├── main_emodb_male.py  
│   │   ├── main_ravdess_female.py  
│   │   ├── main_ravdess_male.py  
│   │   ├── main_shemo_female.py  
│   │   └── main_shemo_male.py  
│   │  
│   └── support  
│       ├── build_features.py  
│       ├── calculations.py  
│       ├── classes_and_adjustments.py  
│       ├── configuration_classes.py  
│       ├── configurations_methods.py  
│       ├── configurations_variables.py  
│       ├── custom_exceptions.py  
│       ├── data_analysis.py  
│       ├── data_cleaning.py  
│       ├── data_loading.py  
│       ├── directory_file_checking.py  
│       ├── models.py  
│       ├── plots_and_charts.py  
│       └── saving_loading.py  
│        
│        
│      
│          
├── backbone_independent(d)  
│   ├── base_store(d)  
│   │   ├── saved_modelconfigs(d)  
│   │   │   ├── cremad(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   ├── emodb(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   ├── ravdess(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   └── shemo(d)  
│   │   │       ├── female(d)   
│   │   │       └── male(d)  
│   │   │  
│   │   ├── saved_models(d)  
│   │   │   ├── cremad(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   ├── emodb(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   ├── ravdess(d)  
│   │   │   │   ├── female(d)   
│   │   │   │   └── male(d)  
│   │   │   └── shemo(d)  
│   │   │       ├── female(d)   
│   │   │       └── male(d)  
│   │   │  
│   │   └── saved_training_metrics_logs(d)  
│   │       ├── cremad(d)  
│   │       │   ├── female(d)   
│   │       │   └── male(d)  
│   │       ├── emodb(d)  
│   │       │   ├── female(d)   
│   │       │   └── male(d)  
│   │       ├── ravdess(d)  
│   │       │   ├── female(d)   
│   │       │   └── male(d)  
│   │       └── shemo(d)  
│   │           ├── female(d)   
│   │           └── male(d)  
│   │  
│   ├── mains-predictions(d)  
│   │   ├── main_prerecorded_upload.py  
│   │   └── main_real_time.py  
│   |  
│   ├── preprediction_audio_store(d)  
│   │   └── uploads(d)  
│   │   
│   └── support  
│       ├── calculations.py  
│       ├── configuration_classes.py  
│       ├── configurations_methods.py  
│       ├── configurations_variables.py  
│       ├── directory_file_checking.py  
│       ├── plotting.py  
│       ├── predict.py  
│       ├── recording_configurations.py  
│       ├── remapping_modules.py  
│       └── saving_loading.py  
│  
└── speech_analysis_raspi(d)  
    ├── base_store(d)  
    │   ├── saved_modelconfigs(d)  
    │   │   ├── cremad(d)  
    │   │   │   ├── female(d)   
    │   │   │   └── male(d)  
    │   │   ├── emodb(d)  
    │   │   │   ├── female(d)   
    │   │   │   └── male(d)  
    │   │   ├── ravdess(d)  
    │   │   │   ├── female(d)   
    │   │   │   └── male(d)  
    │   │   └── shemo(d)  
    │   │       ├── female(d)   
    │   │       └── male(d)  
    │   │  
    │   ├── saved_models(d)  
    │   │   ├── cremad(d)  
    │   │   │   ├── female(d)   
    │   │   │   └── male(d)  
    │   │   ├── emodb(d)  
    │   │   │   ├── female(d)   
    │   │   │   └── male(d)  
    │   │   ├── ravdess(d)  
    │   │   │   ├── female(d)   
    │   │   │   └── male(d)  
    │   │   └── shemo(d)  
    │   │       ├── female(d)   
    │   │       └── male(d)  
    │   │  
    │   └── saved_training_metrics_logs(d)  
    │       ├── cremad(d)  
    │       │   ├── female(d)   
    │       │   └── male(d)  
    │       ├── emodb(d)  
    │       │   ├── female(d)   
    │       │   └── male(d)  
    │       ├── ravdess(d)  
    │       │   ├── female(d)   
    │       │   └── male(d)  
    │       └── shemo(d)  
    │           ├── female(d)   
    │           └── male(d)  
    │  
    ├── mains-predictions(d)  
    │   ├── __init__.py  
    │   ├── finding_audio_capable_devices.py  
    │   ├── main_prerecorded_time_specified_file_store.py  
    │   ├── main_prerecorded_time_specified_noise.py    
    │   ├── main_prerecorded_upload.py  
    │   └── main_real_time.py  
    |  
    ├── preprediction_audio_store(d)  
    │   ├── noisy_clips(d)  
    │   ├── time_sepcified(d)  
    │   └── uploads(d)  
    │   
    ├── support  
    │   ├── __init__.py  
    │   ├── calculations.py  
    │   ├── configuration_classes.py  
    │   ├── configurations_methods.py  
    │   ├── configurations_variables.py  
    │   ├── directory_file_checking.py  
    │   ├── plotting.py  
    │   ├── prerecorded_predict.py  
    │   ├── realtime_predict.py  
    │   ├── recording_configurations.py  
    │   ├── remapping_modules.py  
    │   └── saving_loading.py  
    │   
    ├──  requirements.txt  
    │  
    └── setup.py  
