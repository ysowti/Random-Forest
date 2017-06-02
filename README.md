# Doximity's Data Scientist Assignment
##### Proprietary and Confidential
------------


You are working for a large marketing firm that targets doctors based on their practice area. Your current campaign is targeting Cardiologists, and you are tasked with coming up with a list of doctors to contact.

You are given a file, **`physicians.csv`**, which contains a list of doctors and their unique specialty. You notice that there are a decent number of doctors with "Cardiology" as a specialty, but there are also quite a few doctors whose specialty is "Unknown". You wonder if any of these doctors might actually be Cardiologists.

You find a public dataset, **`procedures.csv`**, which contains a list of procedures your doctors performed over the past year.
The columns of this dataset are as follows:

* **`physician_id`** unique physician identifier, joins to `id` in **`physicians.csv`**
* **`procedure_code`** unique code representing a procedure
* **`procedure`** description of the procedure performed
* **`number_of_patients`** the number of patients the doctor performed that procedure on over the past year

------------

Using this procedure data, determine whether the pool of Cardiologists can be increased.

Both files can be found in this repository. Please submit all relevant code and write a detailed explanation of your methodology and results. Assume both a data scientist and a product manager will look over your results. We’re a Python shop - feel free to use any libraries you see fit.

------------

## Assignment Instructions

* **DO NOT PUSH TO MASTER**, **DO NOT FORK THE REPOSITORY**
* Clone this repository, create a local git branch and commit your changes to said branch.
* Push branch to GitHub.
* Once work is completed, create a new Pull Request between master and your branch.
* *Upon completion, notify Dean Lucas & Pat Blachly via email (dlucas@doximity.com, pblachly@doximity.com).*
