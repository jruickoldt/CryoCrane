# Tutorial - Automated micrograph evaluation by CryoPike and NyquistPike

Although looking at exposures is fun, it is even better to automate the evaluation. Within CryoCrane you can use two sub-programs for this:

|Program | Purpose | Start button | 
| ------- | ------- |  ------- | 
| NyquistPike | estimates the CTF fit extent based on the powerspectrum | Start powerspectrum signal estimation |
| CryoPike | estimates the quality of the micrograph based on the real image | Predict score (you can specify the model on the dropdown menu). |

Both programs rate the micrograph with a score ranging from 0 to 1. For NyquistPike a score of 1 equals an CTF fit extent to the Nyquist-limit and 0.5 to half the Nyquist-limit (e.g. 4 Å at a pixel size of 1 Å). For CryoPike a score of 1 equals a perfect micrographs, between 0.8 and 1 an amazing micrograph, from 0.7 to 0.8 a good micrograph, from 0.5 to 0.7 a maybe usable micrograph. Below a score of 0.5 micrographs are considered junk. 

The scoring takes around 1 minute per 100 micrographs, when running on a customer-grade CPU. The scoring progress is visualized in the progress bar on the bottom right and in the terminal output. Once the scoring is finished, you will see your exposures appearing in new colors. 

## Visual inspection of the micrograph evaluation

After the prediction has finished, the new option "predicted score" and "estimated powerspectrum signal" will appear in the "Colour by" menu. The predicted CryoPike score will be coloured from violet (0) to yellow (1) and the NyquistPike score (estimated powerspectrum signal) from pink (bad) to blue (good). 

## Update your session

To supervise running data collections, you can press the "Update" button. This will automatically collect all new exposures in the specified directory. If NyquistPike or CryoPike were used beforehand, the scores of the new exposures will be updated as well. Non-updated exposures are marked with a score of -1. 

Well, you can keep clicking on the "Update" button. But you can also tick the "Auto-update every 5 min" box. While the auto-update function is enabled, several alignment/prediction functions are disabled to prevent conflicts in the data base. By unticking "Auto-update every 5 min" box these will be enabled again. 

## Saving and loading your session

This is a good point to save your session. Press the "Save session" button. By default, all sessions are stored in the ./sessions folder. The stored csv file is human-readable and contains all data about the exposures including the predicted scores. "Load session" will prompt you to the ./sessions folder and you can simply load older sessions without any hustle.