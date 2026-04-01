# InsightOps-AI-Automated-Data-Intelligence-Decision-Support-Platform

InsightOps AI is an end-to-end data science accelerator that transforms raw datasets into executive-level strategic reports. By leveraging a multi-agent architecture, the system automates data cleaning, performs competitive AutoML model selection, generates interactive visualizations, and synthesizes a professional PDF audit.

🌟 Key Features
Multi-Agent Intelligence: Utilizes specialized agents for data preprocessing, visualization, and narrative synthesis.

Universal AutoML Pipeline: Automatically competes XGBoost, RandomForest, LightGBM, and Linear/Logistic Regression to find the optimal logic for your specific data.

Real-time Progress Streaming: Uses Server-Sent Events (SSE) to provide live updates to the UI during heavy computation.

Executive PDF Reporting: Generates a distribution-ready PDF featuring feature importance, model metrics, and AI-driven insights.

Sanitized Visualization Engine: Automatically handles complex data types (like Dates) to produce clean, readable charts without file-system conflicts.

%% InsightOps AI Multi-Agent System Architecture

graph TD
    %% Define Node Styles
    classDef web fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef agent fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:10,ry:10;
    classDef pipe fill:#fff3e0,stroke:#ef6c00,stroke-width:1px;
    classDef report fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px;

    %% 1. Web Layer
    User --"(Upload CSV, Select Target)"--> Web_UI
    Web_UI --"(Start Strategic Audit)"--> API_Server:::web

    subgraph Orchestration
        %% The main logic flows into App.py, which acts as the supervisor
        API_Server --"(Session ID & Files)"--> App.py:::agent
    end

    subgraph Data_Preparation
        %% These files handle the core ingestion and cleaning
        App.py --"1. Prepare"--> InspectData:::pipe
        InspectData --> Joiner:::pipe
        Joiner --> FeatureEng:::pipe
        FeatureEng --"(Cleaned Data)"--> PreprocessAgent:::agent
    
        %% Specialized detection
        PreprocessAgent --"Detect Target"--> TargetDetector
        TargetDetector --"Detect Problem Type"--> ProblemDetector
    end

    subgraph AutoML_Pipeline
        %% The core modeling competitive loop
        PreprocessAgent --"(Processed Data)"--> AutomlPipe:::pipe
    
        AutomlPipe --"競 爭"--> ModelSelector
        ModelSelector --> ModelLib
        ModelLib --"Competition"--> TrainModels
    
        %% Tuning of the winner
        TrainModels --"Tune Winner"--> Tuning
        Tuning --"Explain Logic"--> Explainer
    
        %% Output of winning model & scores
        Explainer --"(Best Model & Score)"--> App.py
    end

    subgraph Insight_Generation
        %% Post-processing agents
        PreprocessAgent --"EDA"--> EDA
        EDA --"Generate Charts"--> VizAgent:::agent
        VizAgent --"Agnostic Viz"--> Visualization
    
        %% Narrative synthesis
        VizAgent --> AnalystAgent:::agent
        AnalystAgent --"(AI Narrative & Viz List)"--> App.py
    end

    subgraph Report_Output
        %% The finalized delivery
        App.py --"(All Data)"--> ReportGen:::report
        ReportGen --"Distribution-Ready PDF"--> API_Server
    end


📊 How it Works
Data Ingestion: Upload any .csv dataset.

Target Selection: Define the column you wish to predict (Classification or Regression) or else you can depend on the AI to detect the suitable target for you.

Autonomous Processing:

The Preprocessing Agent handles missing values and categorical encoding.

The AutoML Pipeline trains multiple models and identifies the "Winning Logic."

The Visualizer Agent generates feature correlation and trend charts.

Insight Synthesis: The system generates an Executive Summary and a downloadable PDF report.


InsightOps AI: Functional Modular Architecture

┌─────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│ User/CSV│───▶│   Orchestrator│───▶│ Data Agents  │───▶│Model/Pipeline │
└─────────┘    │  (App.py)   │    └──────────────┘    └───────────────┘
               └─────────────┘          │                    │
                      ▲                 ▼                    ▼
                      │         ┌──────────────┐    ┌───────────────┐
                      └─────────│ Insight Agents│◀───│ Final Report  │
                                └──────────────┘    └───────────────┘
                                                       (index.html)
