# ac-mlops-pipeline
This project is an mlops stacks implementation , its a mono-repo aproach, this document captures the implementation details.

The "Getting Started" docs can be found at https://learn.microsoft.com/azure/databricks/dev-tools/bundles/mlops-stacks.

## Table of contents
* [Code structure](#code-structure): structure of this project.

* [Configure your ML pipeline](#configure-your-ml-pipeline): making and testing ML code changes on Databricks or your local machine.

* [Iterating on ML code](#iterating-on-ml-code): making and testing ML code changes on Databricks or your local machine.
* [Next steps](#next-steps)

This directory contains an ML project based on the default
[Databricks MLOps Stacks](https://github.com/databricks/mlops-stacks),
defining a production-grade ML pipeline for automated retraining and batch inference of an ML model on tabular data.

## Code structure
This project contains the following components:

| Component                  | Description                                                                                                                                                                                                                                                                                                                                             |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ML Code                    | ML project code, with unit tested Python modules and notebooks                                                                                                                                                                                                                                                                                  |
| ML Resources as Code | ML pipeline resources (training and batch inference jobs with schedules, etc) configured and deployed through [databricks CLI bundles](https://learn.microsoft.com/azure/databricks/dev-tools/cli/bundle-cli)                                                                                              |

contained in the following files:

```
ac-mlops-pipeline        <- Root directory. Both monorepo and polyrepo are supported.
│
├── clv_mlops_pipeline       <- Contains python code, notebooks and ML resources related to one ML project. 
│   │
│   ├── requirements.txt        <- Specifies Python dependencies for ML code (for example: model training, batch inference).
│   │
│   ├── databricks.yml          <- databricks.yml is the root bundle file for the ML project that can be loaded by databricks CLI bundles. It defines the bundle name, workspace URL and resource config component to be included.
│   │
│   ├── training                <- Training folder contains Notebook that trains and registers the model with feature store support.
│   │
│   ├── feature_engineering     <- Feature computation code (Python modules) that implements the feature transforms.
│   │                              The output of these transforms get persisted as Feature Store tables. Most development
│   │                              work happens here.
│   │
│   ├── evaluation              <-  Evaluate the model with the given metric to take decisions on deployment
│   │
│   ├── monitoring              <- Model monitoring, feature monitoring, etc. (Optional - Added a provision for future)
│   │
│   ├── inference              <- Deployment and Batch inference workflows
│   │   │
│   │   ├── batch_inference     <- Batch inference code that will run as part of scheduled workflow.
│   │
│   │
│   ├── tests                   <- Unit tests for the ML project, including the modules under `features`. (Optional - Added a provision for future)
│   │
│   ├── resources               <- ML resource (ML jobs, MLflow models) config definitions expressed as code, across dev/staging/prod/test.
│       │
│       ├── model-workflow-resource.yml                <- (EXAMPLE) ML resource config definition for model training, validation, deployment workflow
│       │
│       ├── batch-inference-workflow-resource.yml      <- (EXAMPLE) ML resource config definition for batch inference workflow
│       │
│       ├── clv-lifetimevalue-bucket-classifier-DE.yml  <- E2E use-case implementation for lifetime value classification for the country "DE"
|       | 
│       ├── clv-repurchase-classifier-DE.yml           <- E2E use-case implementation for repurchase  classification for the country "DE"
│       │
│       ├── ml-artifacts-resource.yml                  <-(EXAMPLE) ML resource config definition for model and experiment
│       │
│       ├── monitoring-resource.yml           <- (EXAMPLE) ML resource config definition for quality monitoring workflow
```


## Configure your ML pipeline

The sample ML code consists of the following:

* Data extration notebooks helps to extract the data required for feature engieering , user need to implement their own logic based on the source
* Feature computation modules under `feature_engineering` folder. 
These module contains features logic that can be used to generate and populate tables in Feature Store.
There is `compute_features_repurchase_fn` method which is implemented to generate the features from the existing delta table sources
(each column being a separate feature) 
The output dataframe will be persisted in a feature store and for every run feature gets refreshed thru Truncate and load approach 
See the example modules' documentation for more information.
* Python unit tests for feature computation modules in `tests/feature_engineering` folder. - Optional , can be extended for future
* Feature engineering notebook, `feature_engineering/notebooks/GenerateAndWriteFeatures.py`, that reads input dataframes, dynamically loads feature computation modules, executes their `compute_features_repurchase_fn` method and writes the outputs to a Feature Store table (creating it if missing).
* Data segregation notebook helps to segreagate the feature dataframe to X_train,X_test, Y_train and Y_test , in non-prod environment the segregated data would be 25% of the overall data , complete data is used only in the prod environment
* Training notebook that trains a model by creating a training dataset using the Feature Store client.
* Model evaluation helps to evaluate and register the model to Unity catalog 
* Batch inference notebooks that deploy and use the trained model. 

## Configure your Common utils 

AC-MLOPS-PIPELINE 
 - common      => This directory is outside of bundle , so it can be used across multiple use-cases in mono-repo
  - utils.y    => Utils method which can be used across multiple use-cases/models 
  - global.yml => Global configs - repo level
  - clv        => Use-case level config directory
    - lifetime_value_bucket_classifier => model level config directory for lifetime value model
      - common.yml => common configs which can be used across multiple country
      - DE.yml     => country specific config
    - repurchase_classifier => model level config directory for repurchase classifier model
      - common.yml => common configs which can be used across multiple country
      - DE.yml     => country specific config


To adapt this  framework for your use case, implement your own modules, specify the required configs as explained in each notebook of different steps. Refer to the code for detailed comments on every notebook for code level implmentation details.


## Iterating on ML code

### Deploy ML code and resources to dev workspace using Bundles

Refer to [Local development and dev workspace](./resources/README.md#local-development-and-dev-workspace)
to use databricks CLI bundles to deploy ML code together with ML resource configs to dev workspace.

This will allow you to develop locally and use databricks CLI bundles to deploy to your dev workspace to test out code and config changes.

### Develop on Databricks using Databricks Repos

#### Prerequisites
You'll need:
* Access to run commands on a cluster running Databricks Runtime ML version 11.0 or above in your dev Databricks workspace
* To set up [Databricks Repos](https://learn.microsoft.com/azure/databricks/repos/index): see instructions below

#### Configuring Databricks Repos
To use Repos, [set up git integration](https://learn.microsoft.com/azure/databricks/repos/repos-setup) in your dev workspace.

If the current project has already been pushed to a hosted Git repo, follow the
[UI workflow](https://learn.microsoft.com/azure/databricks/repos/git-operations-with-repos#add-a-repo-and-connect-remotely-later)
to clone it into your dev workspace and iterate. 

Otherwise, e.g. if iterating on ML code for a new project, follow the steps below :
* Follow the [UI workflow](https://learn.microsoft.com/azure/databricks/repos/git-operations-with-repos#add-a-repo-and-connect-remotely-later)
  for creating a repo, but uncheck the "Create repo by cloning a Git repository" checkbox.
* Install the `dbx` CLI via `pip install --upgrade dbx`
* Run `databricks configure --profile ac-mlops-pipeline-dev --token --host <your-dev-workspace-url>`, passing the URL of your dev workspace.
  This should prompt you to enter an API token
* [Create a personal access token](https://learn.microsoft.com/azure/databricks/dev-tools/auth/pat)
  in your dev workspace and paste it into the prompt from the previous step
* From within the root directory of the current project, use the [dbx sync](https://dbx.readthedocs.io/en/latest/guides/python/devloop/mixed/#using-dbx-sync-repo-for-local-to-repo-synchronization) tool to copy code files from your local machine into the Repo by running
  `dbx sync repo --profile ac-mlops-pipeline-dev --source . --dest-repo your-repo-name`, where `your-repo-name` should be the last segment of the full repo name (`/Repos/username/your-repo-name`)


### Develop locally

You can iterate on the feature transform modules locally in your favorite IDE before running them on Databricks.  

#### Running code on Databricks
You can iterate on ML code by running the provided `feature_engineering/notebooks/GenerateAndWriteFeatures.py` notebook on Databricks using
[Repos](https://learn.microsoft.com/azure/databricks/repos/index). This notebook drives execution of
the feature transforms code defined under ``features``. You can use multiple browser tabs to edit
logic in `features` and run the feature engineering pipeline in the `GenerateAndWriteFeatures.py` notebook.

#### Prerequisites
* Python 3.8+
* Install feature engineering code and test dependencies via `pip install -I -r requirements.txt` from project root directory.
* The features transform code uses PySpark and brings up a local Spark instance for testing, so [Java (version 8 and later) is required](https://spark.apache.org/docs/latest/#downloading).
* Access to UC catalog and schema
We expect a catalog to exist with the name of the deployment target by default. 
For example, if the deployment target is dev, we expect a catalog named dev to exist in the workspace. 
If you want to use different catalog names, please update the target names declared in the [databricks.yml](./databricks.yml) file.
If changing the staging, prod, or test deployment targets, you'll also need to update the workflows located in the .github/workflows directory.

For the ML training job, you must have permissions to read the input Delta table and create experiment and models. 
i.e. for each environment:
- USE_CATALOG
- USE_SCHEMA
- MODIFY
- CREATE_MODEL
- CREATE_TABLE

For the batch inference job, you must have permissions to read input Delta table and modify the output Delta table. 
i.e. for each environment
- USAGE permissions for the catalog and schema of the input and output table.
- SELECT permission for the input table.
- MODIFY permission for the output table if it pre-dates your job.

#### Run unit tests
You can run unit tests for your ML code via `pytest tests`.



## Next Steps

When you're satisfied with initial ML experimentation (e.g. validated that a model with reasonable performance can be trained on your dataset) and ready to deploy production training/inference pipelines, ask your ops team to set up CI/CD for the current ML project if they haven't already. CI/CD can be set up as part of the

MLOps Stacks initialization even if it was skipped in this case, or this project can be added to a repo setup with CI/CD already, following the directions under "Setting up CI/CD" in the repo root directory README.

To add CI/CD to this repo:
 1. Run `databricks bundle init mlops-stacks` via the Databricks CLI
 2. Select the option to only initialize `CICD_Only`
 3. Provide the root directory of this project and answer the subsequent prompts

More details can be found on the homepage [MLOps Stacks README](https://github.com/databricks/mlops-stacks/blob/main/README.md).

For this project CI-CD is already setup and the two important CI-CD pipelines are ,

.azure/devops-pipelines/ac-mlops-pipeline-bundle-cicd.yml 
  1. Multi-Stage pipeline 
  2. Stage 1 - triggers when the PR is raised to main branch 
  3. Stage 2 - triggers when the PR is merged to main branch
  4. Stage 3 - triggers when the PR is merged to release branch
The steps inside the pipeline can be customized by the user based on the requirements 

.azure/devops-pipelines/ac-mlops-pipeline-tests.yml
1. Triggers whenever the code is pushed to main branch
2. Runs unit and Integration tests which is E2E 

Databricks CLI version v0.230.0 is used to leverage the sync feature to sync the common directory which is outside of the bundle package.
Check the respective pipelines to understand the script level implementation details