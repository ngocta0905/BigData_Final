from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import string


# Clean job posting dataset
def clean_job_dataset():
    path = 'datasets/naukri_com-job_sample.csv'
    
    # columns = ["labels", "text"]
    df_train = pd.read_csv(path, sep=',', skipinitialspace=True)
    # Dataset.from_pandas(df_train)
    # print(df_train[:10])
    # count = 0
    # for line in df_train.values:
    #     if(count < 10):
    #         print(line[4])  
    #         count = count + 1


    #remove empty rows
    df_train['jobdescription'].replace('', np.nan, inplace=True)
    df_train.dropna(subset=['jobdescription'], inplace=True)

    #write cleaned data to new file
    with open('datasets/cleanedData.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        # writer.writerow(['company'] + ['education'] + ['experience'] + ['industry'] + ['jobdescription'] + ['jobid'] + ['joblocation_address'] + ['jobtitle'] + ['numberofpositions'] + ['payrate'] + ['postdate'] + ['site_name'] + ['skills'] + ['uniq_id'])
        writer.writerow(['company'] + ['education'] + ['experience'] + ['industry'] + ['jobdescription'] + ['joblocation_address'] + ['jobtitle'] + ['payrate'] + ['skills'])
        count = 0
        for line in df_train.values:
            if(count >= 0):
                jobDescription = line[4]
                jobDescription = jobDescription[41:]
                writer.writerow([line[0]] + [line[1]] + [line[2]] + [line[3]] + [jobDescription] + [line[6]] + [line[7]] + [line[9]] + [line[12]])#skip 5, 8, 10, 11, 13
            count = count + 1



# Clean the second Resume dataset
def clean_resume_dataset_2():
    path = 'datasets/Resume.csv'
    df_train = pd.read_csv(path, sep=',', skipinitialspace=True,)

    #write cleaned data to new file
    with open('datasets/cleanedData2.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Resume_str'] + ['Category '])
        count = 0
        for line in df_train.values:
            if(count >= 0):
                writer.writerow([line[0]] + [line[1]] + [line[3]])
            count = count + 1
    

# Merging two resume datasets
def mergeDatasets():
    resume1Path = 'datasets/UpdatedResumeDataSet.csv'
    resume2Path = 'datasets/cleanedData2.csv'
    resume1 = pd.read_csv(resume1Path, sep=',', skipinitialspace=True)
    resume2 = pd.read_csv(resume2Path, sep=',', skipinitialspace=True)

    #write cleaned data to new file
    with open('datasets/combinedResumes.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Job Title'] + ['Job Description'])
        count = 0
        for line in resume1.values:
            if(count > 0):
                writer.writerow([line[0]] + [line[1]])
            count = count + 1
        count = 0
        for line in resume2.values:
            if(count > 0):
                writer.writerow([line[1]] + [line[0]])
            count = count + 1



# Remove weird characters from the dataset (DOES NOT WORK)
def clean_resumes():
    path = 'datasets/combinedResumes.csv'
    df_train = pd.read_csv(path, sep=',', skipinitialspace=True)

    # df_train.columns = df_train.columns.str.replace('[#,@,&,Ã,¢,Â,€,â,]', '')
    # df_train['Job Description'].strings.str.replace('\W', '')
    # df_train.str.replace('\W', '')
    df_train.columns = df_train.columns.str.replace('\W', '')
    # df_train.columns = df_train.columns.str.replace(r'[^\x00-\x7F]+', '')
    printable = set(string.printable)
    # df_train[1] = df_train[1].apply(lambda row: ''.join(filter(lambda x: x in printable, row)))
    # df_train.columns = re.sub('[^a-zA-Z0-9 \n\.]', ' ', df_train.columns)
    df_train.columns = df_train.columns.str.replace('•', '')



def process(skill):
    # Jobs
    jobPath = 'datasets/cleanedData.csv'
    # jobs = pd.read_csv(jobPath, sep=',', skipinitialspace=True, quoting=csv.QUOTE_NONE, error_bad_lines=False)
    jobs = pd.read_csv(jobPath, sep=',', skipinitialspace=True)
    jobs = jobs.append({'jobdescription': skill}, ignore_index = True) # Add the input job description/ skill to the dataset
    tfidf = TfidfVectorizer(stop_words='english')
    jobs['jobdescription'] = jobs['jobdescription'].fillna('')
    tfidf_matrix = tfidf.fit_transform(jobs['jobdescription'])


    # Resume
    resumePath = 'datasets/combinedResumes.csv'
    # resume = pd.read_csv(resumePath, sep=',', skipinitialspace=True, lineterminator='\n', header=None, error_bad_lines=False)
    resume = pd.read_csv(resumePath, sep=',', skipinitialspace=True)
    tfidf2 = TfidfVectorizer(stop_words='english')
    resume['Job Description'] = resume['Job Description'].fillna('')
    tfidf_matrix2 = tfidf2.fit_transform(resume['Job Description'])

    # Make the Resume matrix the same size as job matrix
    tfidf_matrix3 = np.zeros((21996, 57258))
    xypos = (0,0)
    x, y = xypos
    ysize, xsize = tfidf_matrix2.shape
    xmax, ymax = (x + xsize), (y + ysize)
    tfidf_matrix3[y:ymax, x:xmax] += tfidf_matrix2

    # cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix3)



    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    #Construct a reverse map of indices and job titles
    indices = pd.Series(jobs.index, index=jobs['jobdescription'])

    # Function that takes in job title as input and outputs most similar jobs
    def get_recommendations(jobDes, cosine_sim=cosine_sim):
        idx = indices[jobDes]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        print(sim_scores)
        job_indices = [i[0] for i in sim_scores]
        print("indices " , job_indices)
        return jobs.iloc[job_indices]

    print(get_recommendations(skill))




if __name__ == '__main__':
    # clean_job_dataset()# Clean the job posting dataset
    # clean_resume_dataset_2()# Clean the second Resume dataset
    # mergeDatasets() #Merging two resume datasets
    # clean_resumes()
    skill = 'HR ASSISTANT INTERN       Summary     New graduate seeking work as a Counselor able to facilitate both individual and group therapy sessions to help participants overcome obstacles. Detail-oriented with superior interpersonal skills.        Skills          High energy  Sound judgment   Compassionate  Conflict resolution training  Exceptional problem solver  Excellent communication skills        Excellent writing skills  Customer service skills  Proficiency in Microsoft Excel, Word, PowerPoint and the Internet            Experience      HR Assistant Intern  ,   Company Name  ,   February 2016  -  March 2016    City  ,   State     Provide administrative support to the Human Resources Director.  Verify I-9 documentation for new hires.  Submit the online investigation requests and assists with new employee background checks.  Update HR spreadsheet with employee change requests and process paperwork.         Owner, Operator  ,   Company Name  ,   August 2012  -  Current    City  ,   State     Managed fashion retail store independently.  Provided professional support to staff.  Assisted retail store in exhibiting innovative products.         Preservation Technician I  ,   Company Name  ,   October 2004  -  May 2013    City  ,   State     Responsible for the assembly of fabricated phase boxes, portfolios and custom enclosures for protecting historic and fragile library materials.  Performed archival sound repairs for books and pamphlets which included: rebinding books in the original covers (recasing).  Prepared and submitted books for additional processing at the bindery.  Designed complex enclosures for special projects.         Education and Training      Bachelors of Art    Organizational Leadership  ,   ,   Cleveland State University  ,           April 2018   Organizational Leadership       Associate Degree      Bryant & Stratton College  ,   ,   City    State      April, 2016          EMT Certification      Cuyahoga Community College  ,   ,   City    State      2003          Skills    administrative support, repairs, spreadsheet'     
    process(skill)