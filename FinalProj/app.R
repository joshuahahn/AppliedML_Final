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
library(rsconnect)


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
                 helpText("Training Data"),
                 dataTableOutput("training_preview"),
                 helpText("Testing Data"),
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
                 plotOutput("hyperparameter_results")
               )
             )
    ),
    
    tabPanel("Model Building",
             titlePanel("Model Selection & Hyperparameter Tuning"),
             sidebarLayout(
               sidebarPanel(
                 selectInput("modelType_final", "Model:", choices = c("Random Forest (RF)", 
                                                                "Support Vector Machine (SVM)", 
                                                                "Extreme Gradient Boosting (XGBoost)")),
                 
                 selectInput("resampling_final", "Resampling Method:", choices=c("Cross-validation (CV)",
                                                                           "Repeated Cross-validation (Repeated CV)")),
                 
                 # CV and RepeatedCV parameters
                 numericInput("cv_value_final",
                              "Number of cross-validation folds:",
                              value=1,
                              min=1),
                 conditionalPanel(condition = "input.resampling_final == 'Repeated Cross-validation (Repeated CV)'",
                                  numericInput("rcv_value_final",
                                               "Number of repeats for repeated cross-validation:",
                                               value=1,
                                               min=1)
                 ),
                 
                 # RF parameters
                 conditionalPanel(condition = "input.modelType_final == 'Random Forest (RF)'",
                                  numericInput("mtry_final", "Number of features to select at each split (mtry)",
                                               value=0,
                                               min=0),
                                  numericInput("min_node_final", "Minimum number of observations in terminal nodes (min_node):",
                                               value=0,
                                               min=0)),
                 
                 # SVM parameters
                 conditionalPanel(condition = "input.modelType_final == 'Support Vector Machine (SVM)'",
                                  numericInput("c_param_final", "C Parameter",
                                               value=0,
                                               min=0),
                                  selectInput("svm_model_final", "Model:", choices = c("Linear", 
                                                                                       "Polynomial"))),
                 
                 # XGBoost Paramters
                 conditionalPanel(condition = "input.modelType_final == 'Extreme Gradient Boosting (XGBoost)'",
                                  numericInput("max_depth_final", "Maximum depth (max_depth)",
                                               value=0,
                                               min=0),
                                  numericInput("eta_final", "Learning rate (eta)",
                                               value=0,
                                               min=0),
                                  numericInput("subsample_final", "Subsample proportion (subsample)",
                                               value=0,
                                               min=0),
                                  numericInput("min_child_weight_final", "Minimum Child Weight (min_child_weight)",
                                               value=0,
                                               min=0)),
                 
                 # Submit button
                 actionButton("submit_final", "Tune Parameters"),
                 
               ),
               mainPanel(
                 #textOutput("final_results")
                 verbatimTextOutput("final_results")
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
      resample <- trainControl(method = "repeatedcv",
                               number = cv,
                               repeats = rcv,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)
    }
    
    hyper_grid <- expand.grid(mtry=mtry_input,
                              splitrule = c("gini", "extratrees"),
                              min.node.size=min_node_input)
    
    rf_fit <- train(as.formula(paste(target, "~ .")), data=train_data,
                    method="ranger",
                    trControl=resample,
                    tuneGrid=hyper_grid,
                    verbose = FALSE,
                    metric="ROC",
                    num.trees = 300)

    output$hyperparameter_results <-renderPlot({plot(rf_fit, metric = "ROC", plotType = "level")})
  }
  
  tune_svm <- function(cv, rcv, c_param, train_data, target) {
    if (rcv == "NA") {
      resample <- trainControl(method = "cv",
                               number = cv,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)
    }
    else {
      resample <- trainControl(method = "repeatedcv",
                               number = cv,
                               repeats = rcv,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)
    }
    
    lin_hyper_grid <- expand.grid(C = c(c_param))
    svm_linear_fit <- train(as.formula(paste(target, "~ .")), data=train_data,
                    method="svmLinear",
                    trControl=resample,
                    tuneGrid=lin_hyper_grid,
                    verbose = FALSE,
                    metric="ROC")

    poly_hyper_grid <- expand.grid(C = c(c_param),
                                   degree=c(2,3),
                                   scale=1)
    svm_poly_fit <- train(as.formula(paste(target, "~ .")), data=train_data,
                            method="svmPoly",
                            trControl=resample,
                            tuneGrid=poly_hyper_grid,
                            verbose = FALSE,
                            metric="ROC")

    resamps <- resamples(list(SVM_Linear = svm_linear_fit,
                              SVM_Polynomial = svm_poly_fit))

    output$hyperparameter_results <-renderPlot({bwplot(resamps, layout=c(2,1))})
  }
  
  tune_xgboost <- function(cv, rcv, max_depth, eta, subsample, min_child_weight, train_data, target) {
    if (rcv == "NA") {
      resample <- trainControl(method = "cv",
                               number = cv,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)
    }
    else {
      resample <- trainControl(method = "repeatedcv",
                               number = cv,
                               repeats = rcv,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)
    }
    
    hyper_grid <- expand.grid(nrounds = c(100),
                              max_depth = c(max_depth),
                              eta = c(eta),
                              min_child_weight = c(min_child_weight),
                              subsample = c(subsample),
                              gamma=0,
                              colsample_bytree=1)
    
    xgboost_fit <- train(as.formula(paste(target, "~ .")), data=train_data,
                    method="xgbTree",
                    trControl=resample,
                    tuneGrid=hyper_grid,
                    verbose = FALSE,
                    metric="ROC",
                    verbosity=0)
    
    output$hyperparameter_results <-renderPlot({plot(xgboost_fit, metric = "ROC", plotType = "level")})
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

      #blueprint <- blueprint %>% 
      #  step_dummy(all_nominal_predictors())
      
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
      c_param <- as.numeric(strsplit(input$c_param_input, ",")[[1]])
      
      tune_svm(cv, rcv, c_param, transformed_train, response_variable)
      
    }
    else if (input$modelType == "Extreme Gradient Boosting (XGBoost)") {
      max_depth <- as.numeric(strsplit(input$max_depth_input, ",")[[1]])
      eta <- as.numeric(strsplit(input$eta_input, ",")[[1]])
      subsample <- as.numeric(strsplit(input$subsample_input, ",")[[1]])
      min_child_weight <- as.numeric(strsplit(input$min_child_weight_input, ",")[[1]])
      
      tune_xgboost(cv, rcv, max_depth, eta, subsample, min_child_weight, transformed_train, response_variable)
    }
  })
  
  
  # Final model
  observeEvent(input$submit_final, {
    # Get pre-processing steps from before and apply to data
    
    training_data_nonreactive <- training_data()
    testing_data_nonreactive <- testing_data()
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
    
    #blueprint <- blueprint %>% 
    #  step_dummy(all_nominal_predictors())
    
    blueprint_prep <- prep(blueprint, training = training_data_nonreactive)
    transformed_train <- bake(blueprint_prep, new_data = training_data_nonreactive)
    transformed_test <- bake(blueprint_prep, new_data = testing_data_nonreactive)
    
    # Get sampling parameters
    
    cv <- input$cv_value_final
    if (input$resampling_final == 'Repeated Cross-validation (Repeated CV)') {
      rcv <- input$rcv_value_final
    }
    else {
      rcv <- "NA"
    }
    
    # Get hyperparameters and tune
    
    if (input$modelType == "Random Forest (RF)") {
      mtry <- input$mtry_final
      min_node <- input$min_node_final
      
      final_rf(cv, rcv, mtry, min_node, transformed_train,response_variable, transformed_test)
    } 
    else if (input$modelType == "Support Vector Machine (SVM)") {
      c_param <- input$c_param_final
      svm_model <- input$svm_model_final
      final_svm(cv, rcv, c_param, svm_model, transformed_train, response_variable, transformed_test)
      
    }
    else if (input$modelType == "Extreme Gradient Boosting (XGBoost)") {
      max_depth <- input$max_depth_final
      eta <- input$eta_final
      subsample <- input$subsample_final
      min_child_weight <- input$min_child_weight_final
      
      final_xgboost(cv, rcv, max_depth, eta, subsample, min_child_weight, transformed_train, response_variable, transformed_test)
    }
  })
  
  # Final model functions
  final_rf <- function(cv, rcv, mtry_input, min_node_input, train_data, target, test_data) {
    
    if (rcv == "NA") {
      resample <- trainControl(method = "cv",
                               number = cv,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)
    }
    else {
      resample <- trainControl(method = "repeatedcv",
                               number = cv,
                               repeats = rcv,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)
    }
    
    hyper_grid <- expand.grid(mtry=mtry_input,
                              splitrule = c("gini"),
                              min.node.size=min_node_input)
    
    rf_fit <- train(as.formula(paste(target, "~ .")), data=train_data,
                    method="ranger",
                    trControl=resample,
                    tuneGrid=hyper_grid,
                    verbose = FALSE,
                    metric="ROC",
                    num.trees = 300)
    
    predictions <- predict(rf_fit, newdata = test_data)
    
    results <- confusionMatrix(test_data[[target]], predictions)
    
    output$final_results <- renderPrint({results})
  }
  
  tune_svm <- function(cv, rcv, c_param, type, train_data, target, test_data) {
    if (rcv == "NA") {
      resample <- trainControl(method = "cv",
                               number = cv,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)
    }
    else {
      resample <- trainControl(method = "repeatedcv",
                               number = cv,
                               repeats = rcv,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)
    }
    
    if (type == "Linear") {
      hyper_grid <- expand.grid(C = param)
      svm_fit <- train(as.formula(paste(target, "~ .")), data=train_data,
                              method="svmLinear",
                              trControl=resample,
                              tuneGrid=lin_hyper_grid,
                              verbose = FALSE,
                              metric="ROC")
    }
    else if (type == "Polynomial") {
      hyper_grid <- expand.grid(C = c_param,
                                     degree=2,
                                     scale=1)
      svm_fit <- train(as.formula(paste(target, "~ .")), data=train_data,
                            method="svmPoly",
                            trControl=resample,
                            tuneGrid=poly_hyper_grid,
                            verbose = FALSE,
                            metric="ROC")
    }

    predictions <- predict(svm_fit, newdata= test_data)
    results <- confusionMatrix(test_data[[target]], predictions)
    
    output$final_results <-renderText({results})
  }
  
  tune_xgboost <- function(cv, rcv, max_depth, eta, subsample, min_child_weight, train_data, target, test_data) {
    if (rcv == "NA") {
      resample <- trainControl(method = "cv",
                               number = cv,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)
    }
    else {
      resample <- trainControl(method = "repeatedcv",
                               number = cv,
                               repeats = rcv,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)
    }
    
    hyper_grid <- expand.grid(nrounds = c(100),
                              max_depth = c(max_depth),
                              eta = c(eta),
                              min_child_weight = c(min_child_weight),
                              subsample = c(subsample),
                              gamma=0,
                              colsample_bytree=1)
    
    xgboost_fit <- train(as.formula(paste(target, "~ .")), data=train_data,
                         method="xgbTree",
                         trControl=resample,
                         tuneGrid=hyper_grid,
                         verbose = FALSE,
                         metric="ROC",
                         verbosity=0)
    
    predictions <- predict(xgboost_fit, newdata = test_data)
    results <- confusionMatrix(test_data[[target]], predictions)
    
    output$final_results <-renderText({results})
  }
  
  
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