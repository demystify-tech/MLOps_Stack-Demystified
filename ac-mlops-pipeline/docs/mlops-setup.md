# MLOps Setup Guide
[(back to main README)](../README.md)

## Table of contents
* [Intro](#intro)
* [Create a hosted Git repo](#create-a-hosted-git-repo)
* [Configure CI/CD](#configure-cicd---azure-devops)
* [Merge PR with initial ML code](#merge-a-pr-with-your-initial-ml-code)
* [Create release branch](#create-release-branch)

* [Deploy ML resources and enable production jobs](#deploy-ml-resources-and-enable-production-jobs)
* [Next steps](#next-steps)

## Intro
This page explains how to productionize the current project, setting up CI/CD and
ML resource deployment, and deploying ML training and inference jobs.

After following this guide, data scientists can follow the [ML Pull Request](ml-pull-request.md) guide to make changes to ML code or deployed jobs.

## Create a hosted Git repo
Create a hosted Git repo to store project code, if you haven't already done so. From within the project
directory, initialize Git and add your hosted Git repo as a remote:
```
git init --initial-branch=main
```

```
git remote add upstream <hosted-git-repo-url>
```

Commit the current `README.md` file and other docs to the `main` branch of the repo, to enable forking the repo:
```

git add README.md docs .gitignore ac_mlops_pipeline/resources/README.md
git commit -m "Adding project README"

git push upstream main
```

## Configure CI/CD - Azure DevOps

Azure DevOps Pipelines are defined under `.azure/devops-pipelines`:
- **`deploy-cicd.yml`**:<br>
  - Generates the other CICD pipelines mentioned below for projects<br>
    - Manually triggered with `project_name` as parameter
> Note that this workflow will need to be triggered for each project to set up its CI/CD.  In order to run the `Push CICD Bundle to a New Branch` step in the workflow, the project needs to enable the Build Service to be able to contribute and create a branch for the project,
i.e when the deploy CI/CD pipeline is triggered, the build service that runs this pipeline needs the necessary permissions to be able to push. To do this, go to Project Settings -> Repositories -> Security -> Select <project_name> Build Service under users and 
set "Contribute", "Create Branch", and "Contribute to pull requests" to "Allow".

Project-Specific pipelines:
- **`ac-mlops-pipeline-tests.yml`**:<br>
  - **[CI]** Performs unit and integration tests<br>
    - Triggered on PR to main
- **`ac-mlops-pipeline-bundle-cicd.yml`**:<br>
  - **[CI]** Performs validation of Databricks resources defined under `ac_mlops_pipeline/resources`<br>
    - Triggered on PR to main<br>
  - **[CD]** Deploys Databricks resources to the staging workspace<br>
    - Triggered on merging into main<br>
  - **[CD]** Deploys Databricks resources to the prod workspace<br>
    - Triggered on merging into release
> Note that these workflows are provided as example CI/CD workflows, and can be easily modified to match your preferred CI/CD order of operations.

Within the CI/CD pipelines defined under `.azure/devops-pipelines`, we will be deploying Databricks resources to the defined staging and prod workspaces using the `databricks` CLI. This requires setting up authentication between the `databricks` CLI and Databricks. By default we show how to authenticate with service principals by passing [secret variables from a variable group](https://learn.microsoft.com/en-us/azure/devops/pipelines/scripts/cli/pipeline-variable-group-secret-nonsecret-variables?view=azure-devops). In a production setting it is recommended to either use an [Azure Key Vault](https://learn.microsoft.com/en-us/azure/devops/pipelines/release/azure-key-vault?view=azure-devops&tabs=yaml) to store these secrets, or alternatively use [Azure service connections](https://learn.microsoft.com/en-us/azure/devops/pipelines/library/service-endpoints?view=azure-devops&tabs=yaml). We describe below how you can adapt the project Pipelines to leverage service connections. Let's add these.

```
git add .azure
git commit -m "Adding devops-pipeline files"
git push upstream main
```

### Service principal approach [Default]

By default, we provide Azure Pipelines where authentication is done using service principals.

#### Requirements:
- You must be an account admin to add service principals to the account.
- You must be a Databricks workspace admin in the staging and prod workspaces. Verify that you're an admin by viewing the
  [staging workspace admin console](https://adb-51153483858903.3.azuredatabricks.net#setting/accounts) and
  [prod workspace admin console](https://adb-4300203405439031.11.azuredatabricks.net#setting/accounts). If
  the admin console UI loads instead of the Databricks workspace homepage, you are an admin.
- Permissions to create Azure DevOps Pipelines in your Azure DevOps project. See the following [Azure DevOps prerequisites](https://learn.microsoft.com/en-us/azure/devops/organizations/security/about-permissions).
- Permissions to create Azure DevOps build policies. See the following [prerequisites](https://learn.microsoft.com/azure/devops/repos/git/branch-policies).

#### Steps:

1. Create two service principals - one to be used for deploying and running staging resources, and one to be used for deploying and running production resources. See [here](https://learn.microsoft.com/azure/databricks/administration-guide/users-groups/service-principals) for details on how to create a service principal.
1. [Add the staging and production service principals to your Azure Databricks account](https://learn.microsoft.com/azure/databricks/administration-guide/users-groups/service-principals#add-service-principals-to-your-account-using-the-account-console), and following this add the staging service principal to the staging workspace, and production service principal to the production workspace. See [here](https://learn.microsoft.com/azure/databricks/administration-guide/users-groups/service-principals) for details.
1. Follow ['Get Azure AD tokens for the service principals'](https://learn.microsoft.com/azure/databricks/dev-tools/api/latest/aad/service-prin-aad-token)
to get your service principal credentials (tenant id, application id, and client secret) for both the staging and prod service principals. You will use these credentials as variables in the project Azure Pipelines.

1. Create separate Azure Pipelines under your Azure DevOps project using the ‘Existing Azure Pipelines YAML file’ option. Create one pipeline for each script. See [here](https://docs.microsoft.com/en-us/azure/devops/pipelines/create-first-pipeline) for more details on creating Azure Pipelines.
1. Create a new variable group called `ac-mlops-pipeline variable group` defining the following secret variables, for more details [here](https://learn.microsoft.com/en-us/azure/devops/pipelines/library/variable-groups?view=azure-devops&tabs=classic#create-a-variable-group):

    - `PROD_AZURE_SP_TENANT_ID`: tenant ID for the prod service principal
    - `PROD_AZURE_SP_APPLICATION_ID`: application (client) ID for the prod service principal
    - `PROD_AZURE_SP_CLIENT_SECRET`: client secret for the prod service principal
    - `STAGING_AZURE_SP_TENANT_ID`: tenant ID for the staging service principal
    - `STAGING_AZURE_SP_APPLICATION_ID`: application (client) ID for the staging service principal
    - `STAGING_AZURE_SP_CLIENT_SECRET`: client secret for the prod service principal

  - Ensure that the Azure Pipelines created in the prior step have access to these variables by selecting the name of the pipelines under the 'Pipeline permissions' tab of this variable group.
  - Alternatively you could store these secrets in an [Azure Key Vault](https://learn.microsoft.com/en-us/azure/devops/pipelines/release/key-vault-in-own-project?view=azure-devops&tabs=portal) and link those secrets as variables to be used in the Pipelines.
1. Define [build validation branch policies](https://learn.microsoft.com/en-us/azure/devops/repos/git/branch-policies?view=azure-devops&tabs=browser#build-validation) for the `main` branch using the Azure build pipelines created in step 1. This is required so that any PR changes to the `main` must build successfully before PRs can complete. 
In the case of a monorepo, where there are multiple projects under a single repository, set a [path filter](https://learn.microsoft.com/en-us/azure/devops/repos/git/branch-policies?view=azure-devops&tabs=browser#path-filters) on the build validation policies, such that devops pipelines are only triggered when there are changes to the respective projects (e.g. the path filter would be `/project1/*` to trigger a devops pipeline when changes are made to _only_ files under the `project1` folder).


### Service connection approach [Recommended in production settings]

#### Requirements:
- You must be an Azure account admin to add service principals to the account.
- You must be a Databricks workspace admin in the staging and prod workspaces. Verify that you're an admin by viewing the
  [staging workspace admin console](https://adb-51153483858903.3.azuredatabricks.net#setting/accounts) and
  [prod workspace admin console](https://adb-4300203405439031.11.azuredatabricks.net#setting/accounts). If
  the admin console UI loads instead of the Databricks workspace homepage, you are an admin.
- Permissions to create service connections within an Azure subscription. See the following [prerequisites](https://docs.microsoft.com/azure/devops/pipelines/library/service-endpoints).
- Permissions to create Azure DevOps Pipelines in your Azure DevOps project. See the following [Azure DevOps prerequisites](https://learn.microsoft.com/en-us/azure/devops/organizations/security/about-permissions).
- Permissions to create Azure DevOps build policies. See the following [prerequisites](https://learn.microsoft.com/azure/devops/repos/git/branch-policies).

The ultimate aim of the service connection approach is to use two separate service connections, authenticated with a staging service principal and a production service principal, to deploy and run resources in the respective Azure Databricks workspaces. Taking this approach then negates the need to read client secrets or client IDs from the CI/CD pipelines.

#### Steps:
1. Create two service principals - one to be used for deploying and running staging resources, and one to be used for deploying and running production resources. See [here](https://learn.microsoft.com/azure/databricks/administration-guide/users-groups/service-principals) for details on how to create a service principal.
1. [Add the staging and production service principals to your Azure Databricks account](https://learn.microsoft.com/azure/databricks/administration-guide/users-groups/service-principals#add-service-principals-to-your-account-using-the-account-console), and following this add the staging service principal to the staging workspace, and production service principal to the production workspace. See [here](https://learn.microsoft.com/azure/databricks/administration-guide/users-groups/service-principals) for details.
1. [Create two Azure Resource Manager service connections](https://learn.microsoft.com/en-us/azure/devops/pipelines/library/service-endpoints?view=azure-devops&tabs=yaml#create-a-service-connection) - one to be used to deploy to staging Databricks resources, the other for production resources. Each of these service connections should be authenticated with the respective staging and production service principals created in the prior step.
1. Update pipeline YAML files to use service connections rather than pipeline variables:
   - First, remove any lines where the environment variables are set in tasks in `ac-mlops-pipeline-tests-ci.yml` or `ac-mlops-pipeline-bundle-cicd.yml` files. Specifically, any lines where the following env vars are used: `PROD_AZURE_SP_TENANT_ID`, `PROD_AZURE_SP_APPLICATION_ID`, `PROD_AZURE_SP_CLIENT_SECRET`, `STAGING_AZURE_SP_TENANT_ID`, `STAGING_AZURE_SP_APPLICATION_ID`, `STAGING_AZURE_SP_CLIENT_SECRET`
   - Then, add the following AzureCLI task prior to installing the `databricks` cli in any of the pipeline jobs:

```yaml
# Get Azure Resource Manager variables using service connection
- task: AzureCLI@2
  displayName: 'Extract information from Azure CLI'
  inputs:
    azureSubscription: # TODO: insert SERVICE_CONNECTION_NAME
    addSpnToEnvironment: true
    scriptType: bash
    scriptLocation: inlineScript
    inlineScript: |
      subscription_id=$(az account list --query "[?isDefault].id"|jq -r '.[0]')
      echo "##vso[task.setvariable variable=ARM_CLIENT_ID]${servicePrincipalId}"
      echo "##vso[task.setvariable variable=ARM_CLIENT_SECRET;issecret=true]${servicePrincipalKey}"
      echo "##vso[task.setvariable variable=ARM_TENANT_ID]${tenantId}"
      echo "##vso[task.setvariable variable=ARM_SUBSCRIPTION_ID]${subscription_id}"
```
  > Note that you will have to update this code snippet with the respective service connection names, depending on which Databricks workspace you are deploying resources to.

1. Create separate Azure Pipelines under your Azure DevOps project using the ‘Existing Azure Pipelines YAML file’ option. Create one pipeline for each script. See [here](https://docs.microsoft.com/en-us/azure/devops/pipelines/create-first-pipeline) for more details on creating Azure Pipelines.
1. Define [build validation branch policies](https://learn.microsoft.com/en-us/azure/devops/repos/git/branch-policies?view=azure-devops&tabs=browser#build-validation) for the `main` branch using the Azure build pipelines created in step 1. This is required so that any PR changes to the `main` must build successfully before PRs can complete. 
In the case of a monorepo, where there are multiple projects under a single repository, set a [path filter](https://learn.microsoft.com/en-us/azure/devops/repos/git/branch-policies?view=azure-devops&tabs=browser#path-filters) on the build validation policies, such that devops pipelines are only triggered when there are changes to the respective projects (e.g. the path filter would be `/project1/*` to trigger a devops pipeline when changes are made to _only_ files under the `project1` folder).


### Setting up CI/CD workflows
After setting up authentication for CI/CD, you can now set up CI/CD workflows. We provide a [Deploy CICD workflow](../.azure/devops-pipelines/deploy-cicd.yml) that can be used to generate the other CICD workflows mentioned below for projects. 
This workflow is manually triggered with `project_name` as parameter. This workflow will need to be triggered for each project to set up its set of CI/CD workflows that can be used to deploy ML resources and run ML jobs in the staging and prod workspaces. 
These workflows will be defined under `.azure/devops-pipelines`. After generating these workflows, be sure to go through the above workflow-specific steps again to add the appropriate build branch policies and filters.


## Merge a PR with your initial ML code
Create and push a PR branch adding the ML code to the repository.

```
git checkout -b add-ml-code
git add .
git commit -m "Add ML Code"
git push upstream add-ml-code
```

Open a PR from the newly pushed branch. CI will run to ensure that tests pass
on your initial ML code. Fix tests if needed, then get your PR reviewed and merged.
After the pull request merges, pull the changes back into your local `main`
branch:

```
git checkout main
git pull upstream main
```

## Create release branch
Create and push a release branch called `release` off of the `main` branch of the repository:
```
git checkout -b release main
git push upstream release
git checkout main
```

Your production jobs (model training, batch inference) will pull ML code against this branch, while your staging jobs will pull ML code against the `main` branch. Note that the `main` branch will be the source of truth for ML resource configs and CI/CD workflows.

For future ML code changes, iterate against the `main` branch and regularly deploy your ML code from staging to production by merging code changes from the `main` branch into the `release` branch.

## Deploy ML resources and enable production jobs
Follow the instructions in [ac-mlops-pipeline/resources/README.md](../ac_mlops_pipeline/resources/README.md) to deploy ML resources
and production jobs.

## Next steps
After you configure CI/CD and deploy training & inference pipelines, notify data scientists working
on the current project. They should now be able to follow the
[ML pull request guide](ml-pull-request.md) and 
[ML resource config guide](../ac_mlops_pipeline/resources/README.md)  to propose, test, and deploy
ML code and pipeline changes to production.
