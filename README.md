# Machine Learning, Georgia Tech (OMSCS CS7641, Spring 2018)
# Assignment #1: Supervised Learning

This repository contains generic code to run experiments for the graduate ML course at Georgia Tech (Spring 2018 edition).  The first assignment covered supervised learning, specifically to compare and contrast the behavior of five different supervised learning algorithms over two datasets.  These algorithms were: Decision Trees, Boosted Decision Trees, k-Nearest Neighbors, Support Vector Machines, and Neural Networks.  I selected the MNIST and USPS digits datasets.

The course assigned zero credit to the code written and submitted by students.  Students were granted freedom to write from scratch or copy code from any source.  The scope, depth, and quality of the analysis/report drawn from the experiments determined the student's grade for the assignment.  No restriction was placed against publicly publishing our own code, as long as no analysis was shared.

**Note that if you are a Georgia Tech student taking this course (or any school for that matter), and the course policy explicitly forbids copying code and requires each student to write his/her own code from scratch, please adhere to those rules and your school's honor and integrity code.**

The code in this repository has been sanitized (to the best of my abilities) to avoid showing enough information for a good report.  The settings and parameters used are default or nominally expected basic settings.  Additional parameters and ranges are definitely needed to elicit better experiments and analysis.

MNIST and USPS are well-known in ML.  These datasets are good choices if you have a lot of time.  The datasets have too many records and have high dimensions, making the experiments time-intensive.  A smaller subset is a wise consideration.  It is of course trivial to change the code for any other dataset.

Finally, I am unsure about the absolute correctness of this code.  I did successfuly use an earlier version of this code during my time, refactored that old code a little bit, and re-run the revised code before posting here.  I think it works correctly and consistently, but I did not check exactly.  Also, Python3.  Notwithstanding the odd print tuples, this should work on Python2 but the machine I used to test this was only Python3.  Sorry, too lazy today.
