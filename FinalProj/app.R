# STAT 3106 Applied Machine Learning Final Project
# Joshua Hahn
# Wyatt King
# Haman Abdulmalek

# Imports
library(shiny)
library(shinythemes)
library(shinycssloaders)
library(tidyverse)
library(ggExtra)
library(data.table)
library(caret)
library(tidymodels)
library(DT)
library(randomForest)


data_initial <- read.csv("credit.csv", header = TRUE)
col_names <- names(data_initial)
data_initial <- data_initial[, c(length(col_names), 1:(length(col_names)-1))]
data_initial[,1] <- factor(data_initial[,1])

ui <- fluidPage(
  titlePanel("STAT 3106 Applied Machine Learning Project: Final"),
  navbarPage(
    title = ("STAT 3106"),
    theme = shinytheme("flatly"),
    tabPanel("Overview", icon = icon("info-circle"),
             titlePanel("Overview"),
             helpText("This is an app developed as the final project to STAT 3106: Applied Machine Learning."),
             helpText("You are able to customize a data pre-procesing pipeline, tune hyperparameters, and visualize results."),
             helpText("Authors: Joshua Hahn, Wyatt King, Haman Abdulmalek"),
    ),

    tabPanel("Upload and Preprocess Data", icon = icon("folder-open"),
             titlePanel("Upload Data"),
             sidebarLayout(
               sidebarPanel(
                 selectInput("dataset", "Dataset:", choices = c("Credit Card Approvals", "Upload your own file")),
                 conditionalPanel(condition = "input.dataset == 'Upload your own file'",
                                  helpText("We currently only support classification problems. If you are uploading your own file, the class column must be the first column."),
                                  fileInput("file", "Select your file:",
                                            accept=c("text/csv",
                                                     "text/comma-separated-values,text/plain",
                                                     ".csv"))  
                 ),
                 tabPanel("Spacer", tags$div(style="height: 30px;")),
                 
                 helpText("Select the ratio of the observations you wish to use for training."),
                 sliderInput("validation_split", "Select Validation Split Ratio:",
                             min = 0.01,
                             max =0.99,
                             value = 0.7),
                 tabPanel("Spacer", tags$div(style="height: 30px;")),
                 
                 helpText("Select the preprocessing steps you wish to apply to the dataset."),
                 checkboxGroupInput("preprocessing_steps", label="Preprocessing Steps",
                                    choices = c("Standardize features (scaling & centering)",
                                                "Impute missing features via K-Nearest Neighbors",
                                                "Remove features with zero / near-zero variance",
                                                "Condense feautres via Principal Component Analysis"))
               ),
               mainPanel(
                 dataTableOutput("training_preview"),
                 dataTableOutput("testing_preview")
               )
             )
    ),
    
    tabPanel("Hyperparamter Tuning",
             titlePanel("Model Selection & Hyperparameter Tuning"),
             sidebarLayout(
               sidebarPanel(
                 selectInput("modelType", "Model:", choices = c("Random Forest (RF)", 
                                                                "Support Vector Machine (SVM)", 
                                                                "Extreme Gradient Boosting (XGBoost)")),
                 
                 selectInput("resampling", "Resampling Method:", choices=c("Cross-validation (CV)",
                                                                           "Repeated Cross-validation (Repeated CV)")),
                 
                 # CV and RepeatedCV parameters
                 numericInput("cv_value",
                              "Number of cross-validation folds:",
                              value=1,
                              min=1),
                 conditionalPanel(condition = "input.resampling == 'Repeated Cross-validation (Repeated CV)'",
                                  numericInput("rcv_value",
                                               "Number of repeats for repeated cross-validation:",
                                               value=1,
                                               min=1)
                 ),
                 
                 # RF parameters
                 conditionalPanel(condition = "input.modelType == 'Random Forest (RF)'",
                                  helpText("Please enter paramters as a comma-separated list of numbers."),
                                  textInput("mtry_input", "Number of features to select at each split (mtry)"),
                                  textInput("min_node_input", "Minimum number of observations in terminal nodes (min_node):")),
                 
                 # SVM parameters
                 conditionalPanel(condition = "input.modelType == 'Support Vector Machine (SVM)'",
                                  selectInput("svm_type_input",
                                              "SVM Model Type:",
                                              choices = c("Linear", "Polynomial")),
                                  helpText("Please enter parameters as a comma-separated list of nonnegative numbers."),
                                  textInput("c_param_input", "C Parameter")),
               
               # XGBoost Paramters
               conditionalPanel(condition = "input.modelType == 'Extreme Gradient Boosting (XGBoost)'",
                                helpText("Please enter parameters as a comma-separated list of nonnegative numbers."),
                                textInput("max_depth_input", "Maximum depth (max_depth)"),
                                textInput("eta_input", "Learning rate (eta)"),
                                textInput("subsample_input", "Subsample proportion (subsample)"),
                                textInput("min_child_weight_input", "Minimum Child Weight (min_child_weight)")),
               
               # Submit button
               actionButton("submit", "Tune Parameters"),
               
               ),
               mainPanel(
                 tabsetPanel(
                   tabPanel("Scatterplot", 
                            plotOutput("hyperparameter_results")),
                   tabPanel("Numeric Summary",
                            dataTableOutput("result1"))
                 )
               )
             )
    ),
    
    tabPanel("Fourth Panel",
             titlePanel("Histogram"),
             sidebarLayout(
               sidebarPanel(
                 selectInput("var", "Variable", choices = NULL), 
                 numericInput("bins", "Number of bins", min = 1, max = 50, step = 1, value = 10),
                 radioButtons("color", "Color of bins:",
                              choices = list("Blue" = "blue", "Red" = "red", "Green" = "green"),
                              selected = "blue"),
                 actionButton("click","Submit")
               ),
               
               mainPanel(
                 tabsetPanel(
                   tabPanel("Histogram",
                            plotOutput("plot2"))
                 )
               )
             )
    )
  )
)


# Server
server <- function(input, output, session) {
  
  # Updates file when user uploads a new file (default = carseats.csv)
  File <- reactive({
    if(input$dataset == 'Upload your own file'){
      req(input$file)
      File <- input$file
      df <- data.frame(rbindlist(lapply(File$datapath, fread), use.names = TRUE, fill = TRUE))
      return(df)
    } else {
      return(data_initial)
    } 
  })
  
  
  # Updates training & testing split and data based on slider
  training_data <- reactive({
    if (input$validation_split > 0 && input$validation_split < 1) {
      split_proportions <- initial_split(File(), prop=input$validation_split)
    } else {
      split_proportions <- initial_split(File(), prop=0.7)
    }
    
    return(training(split_proportions))
  })
  
  testing_data <- reactive({
    if (input$validation_split > 0 && input$validation_split < 1) {
      split_proportions <- initial_split(File(), prop=input$validation_split)
    } else {
      split_proportions <- initial_split(File(), prop=0.7)
    }
    
    return (testing(split_proportions))
  })
  
  
  # Updates the viewer on the first tab
  observeEvent(c(training_data(), testing_data(), File()), {
    if (!is.null(File())) {
      output$training_preview <- renderDT({training_data()})
      output$testing_preview <- renderDT({testing_data()})
    }
  })
  
  # Hyperparameter tuning functions
  
  tune_rf <- function(cv, rcv, mtry_input, min_node_input, train_data, target) {
    
    if (rcv == "NA") {
      resample <- trainControl(method = "cv",
                               number = cv,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)
    }
    else {
      resample <- trainControl(method = "rcv",
                               number = cv,
                               repeats = rcv,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)
    }
    
    hyper_grid <- expand.grid(mtry=mtry_input,
                              splitrule = c("gini", "extratrees"),
                              min.node.size=min_node_input)

    print("C")
    
    #print(train_data)
    #print(target)
    #print(train_data[target])
    #(train_data[, target])
    
    
    cols_to_keep <- setdiff(names(train_data), c(target))
    x <- data.matrix(train_data[cols_to_keep])
    y <- data.matrix(train_data[[target]])
    
    
    #rf_fit <- train(as.formula(paste(target, "~ .")),
    rf_fit <- train(x,y, #data=train_data,
                    method="ranger",
                    trControl=resample,
                    tuneGrid=hyper_grid,
                    metric="ROC",
                    num.trees = 200)
    print("D")
    output$hyperparameter_results <-plot(rf_fit, metric = "ROC", plotType = "level")
    print("E")
  }
  
  tune_svm <- function(cv, rcv, svm_type, c_param, train_data, target) {
    
  }
  
  tune_xgboost <- function(cv, rcvmax_depth, eta, subsample, min_child_weight, train_data, target) {
    
  }
  
  observeEvent(input$submit, {
    # Get pre-processing steps from before and apply to data
    
    training_data_nonreactive <- training_data()
    response_variable <- names(training_data_nonreactive)[1]
    
    preprocessing_steps <- input$preprocessing_steps
    blueprint <- recipe(formula=as.formula(paste(response_variable, "~ .")), data = training_data_nonreactive) %>%
      #update_role(response_variable, new_role = "outcome") %>%
      step_string2factor(all_nominal_predictors())
      
      if ("Remove features with zero / near-zero variance" %in% preprocessing_steps) {
        blueprint <- blueprint %>%  
          step_nzv(all_predictors())
      }
      
      if ("Impute missing features via K-Nearest Neighbors" %in% preprocessing_steps) {
        blueprint <- blueprint %>%
          step_impute_knn(all_predictors())
      }
      
      if("Standardize features (scaling & centering)" %in% preprocessing_steps) {
        blueprint <- blueprint %>%
          step_center(all_numeric_predictors()) %>%
          step_scale(all_numeric_predictors())
      }
      
      if("Condense feautres via Principal Component Analysis" %in% preprocessing_steps) {
        blueprint <- blueprint %>%
          step_pca(all_numeric_predictors())
      }

      blueprint <- blueprint %>% 
        step_dummy(all_nominal_predictors())
      
    blueprint_prep <- prep(blueprint, training = training_data_nonreactive)
    transformed_train <- bake(blueprint_prep, new_data = training_data_nonreactive)
    transformed_test <- bake(blueprint_prep, new_data = training_data_nonreactive)
    
    # Get sampling parameters
    
    cv <- input$cv_value
    if (input$resampling == 'Repeated Cross-validation (Repeated CV)') {
      rcv <- input$rcv_value
    }
    else {
      rcv <- "NA"
    }
    
    # Get hyperparameters and tune
    
    if (input$modelType == "Random Forest (RF)") {
      mtry <- as.numeric(strsplit(input$mtry_input, ",")[[1]])
      min_node <- as.numeric(strsplit(input$min_node_input, ",")[[1]])
      
      tune_rf(cv, rcv, mtry, min_node, transformed_train,response_variable)
    } 
    else if (input$modelType == "Support Vector Machine (SVM)") {
      svm_type <- input$svm_type_input
      c_param <- as.numeric(strsplit(input$c_param, ",")[[1]])
      
      tune_svm(cv, rcv, svm_type, c_param, transformed_train, response_variable)
      
    }
    else if (input$modelType == "Extreme Gradient Boosting (XGBoost)") {
      max_depth <- as.numeric(strsplit(input$max_depth_input, ",")[[1]])
      eta <- as.numeric(strsplit(input$eta_input, ",")[[1]])
      subsample <- as.numeric(strsplit(input$subsample_input, ",")[[1]])
      min_child_weight <- as.numeric(strsplit(input$min_child_weight_input, ",")[[1]])
      
      tune_xgboost(cv, rcv, max_depth, eta, subsample, min_child_weight, transformed_train, response_variable)
    }
  })
  
  
  # Hyperparameters for all models
  modelType <- reactive({
    input$modelType
  })
  
  resampling <- reactive({
    input$resampling
  })
  
  cv_value <- reactive({
    input$cv_value
  })
  
  rcv_value <- reactive({
    if (input$resampling == 'Repeated Cross-validation (Repeated CV)') {
      return(input$rcv_value)
    } else {
      return(-1)
    }
  })
  
  observeEvent(File(), {
    updateSelectInput(session, "response",
                      choices = names(File()))
  })
  
  observeEvent(File(), {
    updateSelectInput(session, "explanatory",
                      choices = names(File()))
  }) 
  
  observeEvent(File(), {
    updateSelectInput(session, "var",
                      choices = names(File()))
  })
  
  
  #output$plot1 <- renderPlot({
  #  p = ggplot(data = File(), aes_string(x = input$explanatory, y = input$response)) +
  #    geom_point(alpha = input$shade) +
  #    theme_minimal() 
  #  if(input$marginal) {
  #    p <- ggMarginal(p, type = "histogram")
  #  }
  #  
  #  p
  #})
  
  output$result1 <- renderDataTable({
    summary_data <- summary(File()[[input$response]])
    data.frame(Measure = names(summary_data), Value = as.character(summary_data))
  })
  
  plot2 <- eventReactive(input$click, 
                         ggplot(data = File(), aes_string(x = input$var)) +
                           geom_histogram(binwidth = diff(range(File()[[input$var]]) / input$bins), fill = input$color, color = "black") +
                           labs(x = input$var, y = "Frequency", title = "Histogram") +
                           theme_minimal()
  )
  
  output$plot2 <- renderPlot({
    plot2() 
  })
}





# Run app
shinyApp(ui = ui, server = server)