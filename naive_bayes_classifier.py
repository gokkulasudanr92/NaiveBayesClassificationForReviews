import pandas as pd
import helper as hlp
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

df = pd.read_csv("TrimedData.csv")
df.columns.values
headers = list(df.columns.values)
headers.remove("Comments")
#df = df.drop(headers,axis=1)
df.head()

# The index of headers is and their values
# 1 - Praise
# 2 - Problem
# 3 - Solution
# 4 - Mitigation
# 5 - Neutrality
# 6 - Localization
# 7 - Summary

#Comments Tuple for Praise Data
praise_comments = hlp.extract_data_list_from_csv(df, headers[1]);
praise_train_data = hlp.extract_train_data(praise_comments)
praise_test_data = hlp.extract_test_data(praise_comments)

praise_cl = NaiveBayesClassifier(praise_train_data)

# Compute accuracy
hlp.print_accuracy("Praise", praise_cl, praise_test_data)


#Comments Tuple for Problem Data
problem_comments = hlp.extract_data_list_from_csv(df, headers[2]);
problem_train_data = hlp.extract_train_data(problem_comments)
problem_test_data = hlp.extract_test_data(problem_comments)

problem_cl = NaiveBayesClassifier(problem_train_data)

# Compute accuracy
hlp.print_accuracy("Problem", problem_cl, problem_test_data)

#Comments Tuple for Solution Data
solution_comments = hlp.extract_data_list_from_csv(df, headers[3]);
solution_train_data = hlp.extract_train_data(solution_comments)
solution_test_data = hlp.extract_test_data(solution_comments)

solution_cl = NaiveBayesClassifier(solution_train_data)

# Compute accuracy
hlp.print_accuracy("Solution", solution_cl, solution_test_data)

#Comments Tuple for Mitigation Data
#mitigation_comments = hlp.extract_data_list_from_csv(df, headers[4]);
#mitigation_train_data = hlp.extract_train_data(mitigation_comments)
#mitigation_test_data = hlp.extract_test_data(mitigation_comments)

#mitigation_cl = NaiveBayesClassifier(mitigation_train_data)

# Compute accuracy
#hlp.print_accuracy("Mitigation", mitigation_cl, mitigation_test_data)

#Comments Tuple for Neutrality Data
neutrality_comments = hlp.extract_data_list_from_csv(df, headers[5]);
neutrality_train_data = hlp.extract_train_data(neutrality_comments)
neutrality_test_data = hlp.extract_test_data(neutrality_comments)

neutrality_cl = NaiveBayesClassifier(neutrality_train_data)

# Compute accuracy
hlp.print_accuracy("Neutrality", neutrality_cl, neutrality_test_data)

#Comments Tuple for Localization Data
localization_comments = hlp.extract_data_list_from_csv(df, headers[6]);
localization_train_data = hlp.extract_train_data(localization_comments)
localization_test_data = hlp.extract_test_data(localization_comments)

localization_cl = NaiveBayesClassifier(localization_train_data)

# Compute accuracy
hlp.print_accuracy("Localization", localization_cl, localization_test_data)

#Comments Tuple for Summary Data
#summary_comments = hlp.extract_data_list_from_csv(df, headers[7]);
#summary_train_data = hlp.extract_train_data(summary_comments)
#summary_test_data = hlp.extract_test_data(summary_comments)

#summary_cl = NaiveBayesClassifier(summary_train_data)

# Compute accuracy
#hlp.print_accuracy("Summary", summary_cl, summary_test_data)

