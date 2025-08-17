#!/usr/bin/env python

from multiprocessing.resource_sharer import DupFd
from textwrap import wrap
from tkinter import font
from venv import create
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
import matplotlib.ticker as mtick

matplotlib.pyplot.switch_backend('agg')

###### Bugs per user by function ######
def plot_per_function_data(df, title, ax):
    # Group by Function and Group
    grouped_df = df.groupby(['Function', 'Group', 'UUID'])
    # Get the number of total bugs per user, for each function and group
    grouped_sum = grouped_df['Bug Count'].sum().reset_index()
    # Make a bar plot
    sns.barplot(x='Function', y='Bug Count', hue='Group', data=grouped_sum, ax=ax)
    
    # Rotate tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('')
    ax.set_ylabel('Bug Count', fontsize=20)
    
    # remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set the title
    ax.set_title(title, fontsize=20)

def print_per_function_summary_stats(df):
    # Group by Function and Group
    grouped_df = df.groupby(['Function', 'Group', 'UUID'])
    # Get the number of total bugs per user, for each function and group
    grouped_sum = grouped_df['Bug Count'].sum().reset_index()

    functions = grouped_sum['Function'].unique()
    # N, mean, median, and standard deviation for function in each group
    for func in functions:
        grouped_func = grouped_sum[grouped_sum['Function'] == func]
        print(f'{func}:')
        print(grouped_func.groupby('Group')['Bug Count'].agg(['count', 'mean', 'median', 'std']))
    print()
    # Compute Welch's t-test for each function and group
    for func in functions:
        grouped_func = grouped_sum[grouped_sum['Function'] == func]
        codex_group = grouped_func[grouped_func['Group'] == 'Codex']
        non_codex_group = grouped_func[grouped_func['Group'] == 'Non-Codex']
        if len(codex_group) >= 2 and len(non_codex_group) >= 2:
            t, p = stats.ttest_ind(codex_group['Bug Count'], non_codex_group['Bug Count'], equal_var=False)
            print(f'{func:40s}: t = {t:6.3f} (p = {p:6.3f})')
        else:
            reason = f'Not enough data: Codex: {len(codex_group)}, Non-Codex: {len(non_codex_group)}'
            t, p = np.nan, np.nan
            print(f'{func:40s}: t = {t:6.3f} (p = {p:6.3f}) [{reason}]')

def create_per_function_plot(df, title, filename):
    # Make two subfigures one on top of the other
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    print("Plotting per function data...")
    plot_per_function_data(df, "Number of Bugs Per User by Function and Group", ax1)
    print_per_function_summary_stats(df)
    print("Plotting per function data for passing tests...")
    plot_per_function_data(df[df['Test Passed']], "Number of Bugs Per User by Function and Group (Ignoring Non-Passing Tests)", ax2)
    print_per_function_summary_stats(df[df['Test Passed']])
    # Save the figure, tight layout
    fig.suptitle(title, fontsize=30)
    fig.tight_layout()
    fig.savefig(filename)

###### Bugs per group (Codex vs Non-Codex) ######
def plot_bug_loc_data(df, title, axes):
    # Compute a normalized bug count per line of code for each function and group
    bugs_per_group_uuid = df.groupby(['Group', 'UUID'])['Bug Count'].sum()
    # Quick explanation: we group by Function, Group, and UUID, and look at Function LoC. The
    # Function LoC is duplicated for each UUID, so we use first() to just get the first one.
    # Finally, we group by Group and UUID, and sum to get the sum of LoCs across functions
    # for each user in each group.
    loc_per_group_uuid = df.groupby(['Function', 'Group', 'UUID'])['Function LoC'].first().groupby(['Group','UUID']).sum()
    bugs_per_loc_uuid = (bugs_per_group_uuid / loc_per_group_uuid).reset_index().rename(columns={0: 'Bugs per LoC'})
    # Plot as barplot
    sns.boxplot(x='Group', y='Bugs per LoC', data=bugs_per_loc_uuid, ax=axes)
    axes.set_title(title, fontsize=30)


def print_bug_loc_summary_stats(df):
    # Compute a normalized bug count per line of code for each function and group
    bugs_per_group_uuid = df.groupby(['Group', 'UUID'])['Bug Count'].sum()
    loc_per_group_uuid = df.groupby(['Function', 'Group', 'UUID'])['Function LoC'].first().groupby(['Group','UUID']).sum()
    bugs_per_loc_uuid = (bugs_per_group_uuid / loc_per_group_uuid).reset_index().rename(columns={0: 'Bugs per LoC'})
    # N, mean, median, and standard deviation for each group
    print(bugs_per_loc_uuid.groupby('Group').agg({'Bugs per LoC': ['count', 'mean', 'median', 'std']}))
    # Compute Welch's t-test for the Codex and Non-Codex groups
    print("Welch's t-test for Codex and Non-Codex groups:")
    codex_group = bugs_per_loc_uuid[bugs_per_loc_uuid['Group'] == 'Codex']
    non_codex_group = bugs_per_loc_uuid[bugs_per_loc_uuid['Group'] == 'Non-Codex']
    if len(codex_group) >= 2 and len(non_codex_group) >= 2:
        t, p = stats.ttest_ind(codex_group['Bugs per LoC'], non_codex_group['Bugs per LoC'], equal_var=False)
        if t < 0:
            # If the t-value is negative, then Codex has a lower number of bugs per LoC than Non-Codex
            info = 'Codex has a lower number of bugs per LoC than Non-Codex'
        else:
            info = 'Codex has a higher number of bugs per LoC than Non-Codex'
        print(f'Codex vs Non-Codex: t = {t:6.3f} (p = {p:6.3f}) [{info}]')
    else:
        reason = f'Not enough data: Codex: {len(codex_group)}, Non-Codex: {len(non_codex_group)}'
        t, p = np.nan, np.nan
        print(f'Codex vs Non-Codex: t = {t:6.3f} (p = {p:6.3f}) [{reason}]')

def create_bug_loc_plot(df, title, filename):
    
    fig = plt.figure(figsize=(12,12))
    ax = plt.axes()
    #fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    print(f"Plotting {filename}...")
    #plot_bug_loc_data(df, title, ax)
    
    df = df.copy()
    df['Group'] = df['Group'].replace('Codex', 'Assisted').replace('Non-Codex','Control')

    ## Great Trick: make the group a categorical variable and give it the order to have the correct order
    #df["Group"] = pd.Categorical(df["Group"], ["Control", "Assisted", "Autopilot"])

    bugs_per_group_uuid = df.groupby(['Group', 'UUID'])['Bug Count'].sum()
    # Quick explanation: we group by Function, Group, and UUID, and look at Function LoC. The
    # Function LoC is duplicated for each UUID, so we use first() to just get the first one.
    # Finally, we group by Group and UUID, and sum to get the sum of LoCs across functions
    # for each user in each group.    
    loc_per_group_uuid = df.groupby(['Function', 'Group', 'UUID'])['Function LoC'].first().groupby(['Group','UUID']).sum()
    bugs_per_loc_uuid = (bugs_per_group_uuid / loc_per_group_uuid).reset_index().rename(columns={0: 'Bugs per LoC'})
    
    print(f'filename f{bugs_per_loc_uuid.count()}')

    # Plot as barplot
    sns.boxplot(x='Group', y='Bugs per LoC', notch=False,  data=bugs_per_loc_uuid, showfliers = False, ax=ax, 
                order=['Control', 'Assisted', 'Autopilot'])
    ax.set_title('\n'.join(wrap(title,30)), fontsize=30)
    
    # Added by BDG to create bugs_per_loc_{compiled,passing,severe_compiled,severe_passing}.csv
    bugs_per_loc_uuid.to_csv(filename + '.csv', index=False)

    #fig.suptitle(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    plt.ylabel('Bugs per LoC', fontsize=50, fontweight='bold')
    plt.xlabel('')
    plt.xticks(fontsize=40);
    plt.yticks(fontsize=40);
    
    fig.tight_layout()
    fig.savefig(filename)

def create_cwe_prevalence_plot(df, title, filename, color=False, cwe_key='Immediate CWE'):
    
    print(f"Plotting {filename}...")

    # Make a copy of the df
    df = df.copy()
    # Add nicer descriptions to the CWEs
    with plt.rc_context({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]
    }):
        #df['CWE Name'] = [f'\\textbf{{CWE-{cwe_id}}}' for cwe_id in df[cwe_key].values ]
        # remove the .0 from the CWE ID
        df['CWE Name'] = [f'\\textbf{{CWE-{cwe_id:g}}}' for cwe_id in df[cwe_key].values ]
        #df['CWE Name'] = [f'\\textbf{{CWE-{cwe_id}}}: \n{description}' for cwe_id, description in df[[cwe_key,cwe_key+' Description']].values ]
        
        df = df.copy()
        df['Group'] = df['Group'].replace('Codex', 'Assisted').replace('Non-Codex','Control')

        ## Great Trick: make the group a categorical variable and give it the order to have the correct order
        df["Group"] = pd.Categorical(df["Group"], ["Control", "Assisted", "Autopilot"])
        #df['CWE Name'] = [f'\\textbf{{CWE-{cwe_id:g}}}' for cwe_id in df[cwe_key].values ]
        counts_by_cwe = df.groupby(['Group', 'CWE Name'])['Bug Count'].sum()
        cwe_prevalence = (100*counts_by_cwe / counts_by_cwe.groupby('Group').sum()).reset_index().rename(columns={'Bug Count': 'Prevalence'})
        cwe_prevalence_top = cwe_prevalence.sort_values(by=['Prevalence'], ascending=False).head(24)
        
        fig = plt.figure(figsize=(40,30))
        ax = plt.axes()
        #fig, axes = plt.subplots(1, 2, figsize=(24, 12))



        #sns.barplot(x='CWE Name', y='Prevalence', data=cwe_prevalence[cwe_prevalence['Group'] == 'Codex'], ax=axes[0])
        #sns.barplot(x='CWE Name', y='Prevalence', data=cwe_prevalence[cwe_prevalence['Group'] == 'Non-Codex'], ax=axes[1])
        sns.barplot(x='CWE Name', y='Prevalence', hue='Group', data=cwe_prevalence_top, ax=ax)
        ax.set_title(title, fontsize=80, fontweight='bold')
         # Rotate tick labels top-left to bottom-right
         
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha='left',fontsize=60, wrap=True)
        if color:
            #severe = [1, 2, 3, 4, 5, 6, 8, 9,10, 11, 13, 14, 16, 17, 18,19, 20]
            severe = [0, 2, 4, 5, 6, 7, 8]
            for label in severe:
                #print(label)
                ax.get_xticklabels()[label].set_color("red")
        plt.xlabel('');

        plt.ylabel('');
        plt.yticks(fontsize=60);
        
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.legend(loc='center right', fontsize=70, frameon=False);


        fig.tight_layout()
        fig.savefig(filename)

def create_cwe_kinds_per_user_plot(df, title, filename, cwe_key='Immediate CWE'):
    # This one kind of got away from me. It first groups by Group, CWE, and UUID,
    # and sums bug counts across functions. Then it groups by Group and UUID and
    # counts the distinct CWE categories. Finally, it renames the column to CWE Kinds.
    cwe_kinds_per_user = df.groupby(['Group', cwe_key, 'UUID'])['Bug Count'].sum().reset_index() \
        .groupby(['Group', 'UUID'])[cwe_key].count().reset_index() \
        .rename(columns={cwe_key: 'CWE Kinds'})
    fig, ax = plt.subplots(1,1,figsize=(12, 12))
    sns.boxplot(x='Group', y='CWE Kinds', data=cwe_kinds_per_user, notch=True, ax=ax)
    
    # Set labels and sizes for the groups
    ax.set_xlabel("Group", fontsize=25)
    ax.set_ylabel("CWE Kinds", fontsize=25)
    ax.tick_params(labelsize=20)
    ax.set_title("\n".join(wrap(title)), wrap=True, fontsize=30)
    
    # remove some of the spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()
    t.set_y(1.05) 
    fig.savefig(filename)
    


####################################################################################################
##
## This code creates figures 8 and 9 in the paper. 
## We called it there functionality-embedded.pdf
##
if __name__ == "__main__":
    # Load from CSV
    df = pd.read_csv('bug_finding_flat.csv')

    sns.set_style('whitegrid')

    # Exclude Autopilot from these plots
    df_main = df[df['Group'] != 'Autopilot']
    print(df_main.describe())
    
    
    print("########### Plotting bug count data ###########")
    # # create_per_function_plot(df, 'CWEs Per User by Function and Group', 'figures/bugs_per_function.pdf')
    create_bug_loc_plot(df, '', 'figures/bugs_per_loc_compiled.pdf')
    create_bug_loc_plot(df[df['Test Passed']], '', 'figures/bugs_per_loc_passing.pdf')
    
    df_severe = df[df['CWE Rank'].notna()]
    create_bug_loc_plot(df_severe, '', 'figures/bugs_per_loc_severe_compiled.pdf')    
    create_bug_loc_plot(df_severe[df_severe['Test Passed']], '', 'figures/bugs_per_loc_severe_passing.pdf')

    print(df.head())
    create_cwe_prevalence_plot(df, '', 'figures/cwe_prevalence.pdf', False)
    # create_cwe_kinds_per_user_plot(df, 'CWE kinds per user by Group', 'figures/cwe_kinds_per_user.pdf')
    
    
    # print("########### Plotting severe bug count data ###########")
    # create_per_function_plot(df[df['CWE Rank'].notna()], 'Severe CWEs Per User by Function and Group', 'figures/bugs_per_function_severe.pdf')
    #create_bug_loc_plot(df_main[df['CWE Rank'].notna()], 'Severe CWEs per LoC by Group', 'figures/bugs_per_loc_severe.pdf')
    # create_cwe_prevalence_plot(df[df['CWE Rank'].notna()], 'Severe CWE prevalence by Group', 'figures/cwe_prevalence_severe.pdf')
    # create_cwe_kinds_per_user_plot(df[df['CWE Rank'].notna()], 'Severe CWE kinds per user by Group', 'figures/cwe_kinds_per_user_severe.pdf')
