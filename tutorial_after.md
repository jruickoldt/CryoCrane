# Tutorial - After the data collection

CryoCrane is not only useful during a data collection but also afterwards. You can use it as a tool to document your data collections via CryoCrane sessions, to generate reports or to remove junk micrographs from a data set. 

If your data set has both predicted CryoPike and NyquistPike score you can use the following options. 

## Report generation

After your data collection has finished, CryoCrane has you covered with an automated report generation. Click on the "Generate CryoCrane Report" button. In the appearing pop-up window you can specify the characteristics of your report. The report will be saved in the ./reports folder. 

## Data clean up

---
**NOTE**

The data clean up currently only works for data recorded with EPU.

---

To reduce the load in downstream data processing you can remove junk micrographs from your data set. Click on the "Interactive Data curation" button. This will open a new window. On the left plot the exposures are plotted by their scores. Clicking on a data point will show the respective exposure on the right panel. 

You can now specify thresholds for the detection of junk micrographs. It is good practice to assign these thresholds based on the micrographs at the threshold borders. By changing the logic (AND or OR) you can select micrographs meeting both criteria (AND) or at least one of the thresholds (OR). Assuming that you analysed the summed images, the specify the path to the folder containing the actual movies. Then press "Dry run". CryoCrane will check if it can find all movies. If this is successful, you can press "Delete files". 

---
**WARNING**

The data clean up is irreversible! Take care. 

---