from psychopy import visual, event, core, data, gui, tools, monitors
from psychopy.hardware import keyboard
from glob import glob
import pandas as pd
import numpy as np


# CONSTANTS, EXPERIMENT DETAILS

SCANNING = True  # False for practice, true for the big show


scan_monitor_details = {"name": "Monitor01", "distance": 81, "resolution": [1920, 1080], "width": 56.9}
beh_monitor_details = {"name": "Monitor01", "distance": 70, "resolution": [1920, 1080], "width": 40}

exp_info = {
    "cueTime": 0.5,
    "postCueTime": 1,
    "stimTime": 0.5,
    "initialWait": 3,
    "TR_len": 1.300,
    "ISIs": np.array([3, 3, 4, 5, 6, 8]),  # Need 6 after the 6 images
    "delayTime": 6,
    "ResponseWindow": 3,
    "ITIs": np.array([3, 3, 3, 4, 4, 6, 6, 7, 8]),  # need 9, 1 after each trial
    "finalCoolDown": 20,
    "SetSizes": [1, 2, 3],
    "nPres": 6,
    "keys": ["b", "y"],
    "image_categories": ["fruit", "buildings", "objects"],
    "img_size": 3.5,  # DVA
    "img_loc": [2.75, 2.75],  # how far (in DVA) the stim are from fixation along the x and y axes, respectively
    "bg_color": [0, 0, 0],
}


# HELPER CLASSES AND FUNCTIONS


class masterClock:
    """
    helper class which keeps track of how long each screen should ideally last
    """

    def __init__(self):
        self.clock = core.Clock()
        self.startTime = self.clock.getTime()
        self.idealTime = self.startTime

    def wait(self, time):
        self.idealTime += time
        core.wait(self.idealTime - self.clock.getTime())

    def update_ideal(self, time):
        self.idealTime += time


class Task:

    def __init__(self, monitor_details, exp_info):
        self.monitor_details = monitor_details
        self.exp_info = exp_info

        # initialize the monitor and window
        self.monitor = monitors.Monitor(monitor_details["name"], width=monitor_details["width"], distance=monitor_details["distance"])
        self.monitor.setSizePix(monitor_details["resolution"])
        self.win = visual.Window(monitor=self.monitor, fullscr=True, units="deg", checkTiming=True)
        self.xcenter = int(monitor_details["resolution"][0]/2)
        self.ycenter = int(monitor_details["resolution"][1]/2)

        self.kb = keyboard.Keyboard()

    def get_run_info_and_file(self):
        """
        Get the participant ID, session number, and run number
        Build a filename based on this information, and check to see if a file already exists
        """
        run_info = {"participant ID": 0, "session #": 0, "run #": 0}
        dlg = gui.DlgFromDict(run_info)  # (and from psychopy import gui at top of script)
        if not dlg.OK:
            core.quit()
        run_info["dateStr"] = data.getDateStr()  # will create str of current date/time
        self.run_info = run_info

        self.filename = "sub-{}_ses-{}_run-{}_Scanning-{}_datetime-{}".format(str(run_info['participant ID']).zfill(3),str(run_info['session #']).zfill(2),str(run_info['run #']).zfill(2), SCANNING, run_info['dateStr'])

        old_files = glob("data/" + self.filename.split("_datetime")[0] + "*.tsv")
        if len(old_files):
            warningDlg = gui.Dlg(title="Warning - file exists!")
            warningDlg.addText("A file already exists for this participant/session/run combination. Do you want to continue?")
            warningDlg.show()
            if not warningDlg.OK:
                core.quit()

        if not SCANNING:
            warningDlg = gui.Dlg(title="Warning - in practice mode!")
            warningDlg.addText("You are in practice mode. Do you want to continue?")
            warningDlg.show()
            if not warningDlg.OK:
                core.quit()

        print("Data will be saved to {}".format(self.filename))

    def init_rng(self):
        """
        set a random seed using the info above
        """
        self.rng = np.random.default_rng(int(self.run_info["participant ID"]) * 10000 + int(self.run_info["session #"]) * 1000 + int(self.run_info["run #"])*10 + int(SCANNING))

    # FUNCTIONS FOR INITILIAZING, DRAWING THE STIMULI
    def init_fix(self):
        """
        Initialize the fixation cross
        Based on Thaler et al., 2013
        """

        self.fix_outer = visual.Circle(self.win, lineColor=None, fillColor=[-1, -1, -1], fillColorSpace="rgb", radius=0.25, units="deg", interpolate=True)
        self.fix_cross = visual.ShapeStim(
            self.win,
            vertices=((0, -0.26), (0, 0.26), (0, 0), (-0.26, 0), (0.26, 0)),
            lineWidth=tools.monitorunittools.deg2pix(0.16, self.monitor),
            closeShape=False,
            lineColor=tuple(self.win.color),
            units="deg",
        )
        self.fix_inner = visual.Circle(self.win, lineColor=None, fillColor=[-1, -1, -1], fillColorSpace="rgb", radius=0.075, units="deg", interpolate=True)

    def init_cue(self):
        # make cue out of 4 thin rectangles, pointing to each of the 4 quadrants
        cue_top_left = visual.Rect(self.win, width=0.2, height=0.8, fillColor="black", lineColor="black", ori=135, pos=(-0.5, 0.5), units="deg")
        cue_top_right = visual.Rect(self.win, width=0.2, height=0.8, fillColor="black", lineColor="black", ori=45, pos=(0.5, 0.5), units="deg")
        cue_bottom_right = visual.Rect(self.win, width=0.2, height=0.8, fillColor="black", lineColor="black", ori=-45, pos=(0.5, -0.5), units="deg")
        cue_bottom_left = visual.Rect(self.win, width=0.2, height=0.8, fillColor="black", lineColor="black", ori=-135, pos=(-0.5, -0.5), units="deg")
        self.cue = [cue_top_left, cue_top_right, cue_bottom_right, cue_bottom_left]

        # list the images for each category - there is an a and b version of each image
        self.category_images = {cat: sorted([s.split("/")[-1] for s in glob("images/{}/*".format(cat))]) for cat in self.exp_info["image_categories"]}

        self.category_cues = {}
        for category in self.exp_info["image_categories"]:
            # add text stim on cardinal axes
            top = visual.TextStim(self.win, text=category[0], pos=(1, -.1), color="black", units="norm", alignHoriz='center', alignVert='center', height=.05)
            bottom = visual.TextStim(self.win, text=category[0], pos=(1, .1), color="black", units="norm", alignHoriz='center', alignVert='center', height=.05)
            left = visual.TextStim(self.win, text=category[0], pos=(.925, 0), color="black", units="norm", alignHoriz='center', alignVert='center', height=.05)
            right = visual.TextStim(self.win, text=category[0], pos=(1.075, 0), color="black", units="norm", alignHoriz='center', alignVert='center', height=.05)
            self.category_cues[category] = [top, right, bottom, left]

    def init_stimuli(self):

        self.init_fix()
        self.init_cue()

        # the 4 possible locations where stimuli will be presented
        [x, y] = self.exp_info["img_loc"]
        self.quad_locs = np.array([[-x, y], [x, y], [x, -y], [-x, -y]])  # TL, TR, BR, BL
        self.name_to_quad_dict = {  # giving names to sets of locations (indices for the quad_locs), defined above
            "top": [0, 1],
            "right": [1, 2],
            "bottom": [2, 3],
            "left": [3, 0],
            "all": [0, 1, 2, 3],
        }
        self.spatial_names = [
            "top",
            "right",
            "bottom",
            "left",
            "all",
            "all",
            "all",
            "all",
        ]  # artificially inflating so that there is 50% chance of each spatial set size (2 vs 4)

        sz = self.exp_info["img_size"]
        # build nested dictionary of images
        self.images = {}
        for cat in self.exp_info["image_categories"]:
            self.images[cat] = {}
            for img in self.category_images[cat]:
                self.images[cat][img] = {
                    "a": visual.ImageStim(self.win, image="images/{}/{}/a.jpg".format(cat, img), size=(sz, sz), units="deg"),
                    "b": visual.ImageStim(self.win, image="images/{}/{}/b.jpg".format(cat, img), size=(sz, sz), units="deg"),
                }

        # build similar dictionary of scrambled images
        self.images_scrambled = {}
        for cat in self.exp_info["image_categories"]:
            self.images_scrambled[cat] = {}
            for img in self.category_images[cat]:
                self.images_scrambled[cat][img] = {
                    "a": visual.ImageStim(self.win, image="images_scrambled/{}/{}/a.jpg".format(cat, img), size=(sz, sz), units="deg"),
                    "b": visual.ImageStim(self.win, image="images_scrambled/{}/{}/b.jpg".format(cat, img), size=(sz, sz), units="deg"),
                }

    def draw_fix(self):
        self.fix_outer.draw()
        self.fix_cross.draw()
        self.fix_inner.draw()

    def draw_cue(self, thisTrial):
        target_quads = thisTrial["target_quads"]
        for i, c in enumerate(self.cue):
            if i in target_quads:
                c.fillColor = "red"
            else:
                c.fillColor = "black"
            c.draw()

        # draw the letter cues to indicate which category is being run
        for c in self.category_cues[thisTrial["target_category"]]:
            c.draw()

    def update_trial_locs(self, trial):
        # update the images location, for faster drawing
        for i, img in enumerate(trial["images"]):
            img.pos = self.quad_locs[trial["locations"][i]]

            for j, img in enumerate(trial["placeholder_imgs_pres_{}".format(i)]):
                img.pos = self.quad_locs[trial["placeholder_locations_pres_{}".format(i)][j]]

        trial["test_image"].pos = self.quad_locs[trial["test_location"]]

    def draw_stimuli_pres(self, trial, i):
        trial["images"][i].draw()
        for img in trial["placeholder_imgs_pres_{}".format(i)]:
            img.draw()

    def make_trial(self, set_size, target_category, target_locs, ITI):

        other_categories = [c for c in self.exp_info["image_categories"] if c != target_category]

        target_quads = self.name_to_quad_dict[target_locs]
        nontarget_quads = np.setdiff1d(np.arange(4), target_quads)

        trial = {
            "set_size": set_size,
            "target_category": target_category,
            "target_locs": target_locs,
            "target_quads": target_quads,
            "images": [],
            "image_names": [],
            "AorB": [],
            "locations": [],
            "ISIs": self.rng.permutation(exp_info["ISIs"]),
            "match": self.rng.choice([True, False]),
            "ITI": ITI,
        }

        # PART 1 - GRAB THE TARGET IMAGES

        # make sure not to repeat
        target_images = np.random.choice(self.category_images[target_category], set_size, replace=False)
        ab_choices = np.random.choice(["a", "b"], set_size, replace=True)
        for i in range(set_size):
            trial["images"].append(self.images[target_category][target_images[i]][ab_choices[i]])
            trial["image_names"].append(target_images[i])
            trial["AorB"].append(ab_choices[i])
            trial["locations"].append(self.rng.choice(target_quads))

        # PART 2 - set up non-targets for the remaining presentations
        non_target_images = []
        while len(non_target_images) < (self.exp_info["nPres"] - set_size):
            # randomly grab an image from the other two categories
            attended_bool = self.rng.choice([True, False])

            if (len(target_quads) == 4) or attended_bool:
                trial["locations"].append(self.rng.choice(target_quads))
                curr_categories = other_categories  # can't be a target image if presented at attended location
            else:
                curr_categories = self.exp_info["image_categories"]  # can be any category in unattended locations
                trial["locations"].append(self.rng.choice(nontarget_quads))

            curr_category = self.rng.choice(curr_categories)
            curr_image = self.rng.choice(self.category_images[curr_category])
            while curr_image in non_target_images:  # make sure we don't repeat images
                curr_category = self.rng.choice(curr_categories)
                curr_image = self.rng.choice(self.category_images[curr_category])

            curr_ab = self.rng.choice(["a", "b"])
            # if the current image matches a target image, add the other option (if A then B, if B then A)
            if curr_image in target_images:
                match_idx = np.where(curr_image == target_images)[0][0]
                match_ab = ab_choices[match_idx]
                curr_ab = "b" if match_ab == "a" else "a"

            trial["images"].append(self.images[curr_category][curr_image][curr_ab])
            trial["image_names"].append(curr_image)
            trial["AorB"].append(curr_ab)

            non_target_images.append(curr_image)
            trial["non_target_images"] = non_target_images

        # PART 3 - FIGURE OUT TEST PROBE
        if trial["match"]:
            # grab one of the previous images to be the test image
            test_idx = self.rng.choice(np.arange(set_size))
            trial["test_image"] = trial["images"][test_idx]
            trial["test_image_name"] = trial["image_names"][test_idx]
            trial["test_AorB"] = trial["AorB"][test_idx]
            trial["test_location"] = trial["locations"][test_idx]
            trial["correct_response"] = exp_info["keys"][0]
        else:
            # choose one of the previously defined targets
            target_img = self.rng.choice(target_images)
            target_idx = np.where(target_img == target_images)[0][0]
            old_ab = ab_choices[target_idx]
            new_ab = "a" if old_ab == "b" else "b"

            trial["test_image"] = self.images[target_category][target_img][new_ab]
            trial["test_image_name"] = target_img
            trial["test_AorB"] = new_ab
            trial["test_location"] = trial["locations"][target_idx]  # present at the same location
            trial["correct_response"] = exp_info["keys"][1]

        # PART 4 - resort the images (and locations)
        new_order = self.rng.permutation(np.arange(self.exp_info["nPres"]))
        trial["images"] = [trial["images"][i] for i in new_order]
        trial["image_names"] = [trial["image_names"][i] for i in new_order]
        trial["AorB"] = [trial["AorB"][i] for i in new_order]
        trial["locations"] = [trial["locations"][i] for i in new_order]

        # PART 5 - ADD SCRAMBLED IMAGE PLACEHOLDERS
        n_scrambled = self.exp_info["nPres"] * 3
        n_scrambled_per_category = int(n_scrambled // len(self.exp_info["image_categories"]))
        cat_labels = np.concatenate([[c] * n_scrambled_per_category for c in self.exp_info["image_categories"]])
        placeholder_images = []
        for category in self.exp_info["image_categories"]:
            curr_images = np.random.choice(self.category_images[category], n_scrambled_per_category, replace=False)
            placeholder_images.append(curr_images)
        placeholder_images = list(zip(cat_labels, np.concatenate(placeholder_images)))
        self.rng.shuffle(placeholder_images)  # randomize order
        for i in range(self.exp_info["nPres"]):
            curr_images = placeholder_images[i * 3 : (i + 1) * 3]
            curr_locations = np.setdiff1d(np.arange(4), trial["locations"][i])  # can't be at the old location
            curr_abs = np.random.choice(["a", "b"], 3, replace=True)
            trial["placeholder_img_names_pres_{}".format(i)] = [c[1] for c in curr_images]
            trial["placeholder_locations_pres_{}".format(i)] = curr_locations
            trial["placeholder_AorB_pres_{}".format(i)] = curr_abs
            trial["placeholder_imgs_pres_{}".format(i)] = []
            for j, image in enumerate(curr_images):
                trial["placeholder_imgs_pres_{}".format(i)].append(self.images_scrambled[image[0]][image[1]][curr_abs[j]])

        return trial

    def make_block(self):
        """
        builds a block iterating over combination of set size and target category
        """
        ITIs = self.rng.permutation(exp_info["ITIs"])
        ITI_counter = 0
        trials = []
        for target_category in self.exp_info["image_categories"]:
            for ss in exp_info["SetSizes"]:
                target_locs = self.rng.choice(self.spatial_names)
                trials.append(self.make_trial(ss, target_category, target_locs, ITIs[ITI_counter]))
                ITI_counter += 1

        # shuffle trial order
        self.rng.shuffle(trials)  # this probably is double shuffling the ITIs, but oh well

        return trials

    def run_trial(self, thisTrial):
        """
        Run a single trial

        Note, the order of events is kind of weird. After a screen flips, the next screen is prepared, _then_ a wait period is called.
        This is because the wait function waits based on the _total_ time passed since the last call, so it can absorb the time spent preparing the next screen.
        """

        # cue locations and category
        self.draw_cue(thisTrial)
        self.draw_fix()
        self.win.flip()
        thisTrial["Cue Onset"] = self.masterTimer.clock.getTime()

        # prep ISI, update image locations
        self.draw_fix()
        self.update_trial_locs(thisTrial)

        self.masterTimer.wait(exp_info["cueTime"])

        # ISI
        self.win.flip()

        # prep 1st image
        self.draw_fix()
        self.draw_stimuli_pres(thisTrial, 0)

        self.masterTimer.wait(exp_info["postCueTime"])

        # display 1st image
        self.win.flip()
        thisTrial["Stimulus 1 Onset"] = self.masterTimer.clock.getTime()

        # prep ISI
        self.draw_fix()

        self.masterTimer.wait(exp_info["stimTime"])

        # display ISI
        self.win.flip()

        # prep 2nd image
        self.draw_fix()
        self.draw_stimuli_pres(thisTrial, 1)

        self.masterTimer.wait(thisTrial["ISIs"][0])

        # display 2nd image
        self.win.flip()
        thisTrial["Stimulus 2 Onset"] = self.masterTimer.clock.getTime()

        # prep ISI
        self.draw_fix()

        self.masterTimer.wait(exp_info["stimTime"])

        # ISI
        self.win.flip()

        # prep 3rd image
        self.draw_fix()
        self.draw_stimuli_pres(thisTrial, 2)

        self.masterTimer.wait(thisTrial["ISIs"][1])

        # display 3rd image
        self.win.flip()
        thisTrial["Stimulus 3 Onset"] = self.masterTimer.clock.getTime()

        # prep ISI
        self.draw_fix()

        self.masterTimer.wait(exp_info["stimTime"])

        # ISI
        self.win.flip()

        # prep 4th image
        self.draw_fix()
        self.draw_stimuli_pres(thisTrial, 3)

        self.masterTimer.wait(thisTrial["ISIs"][2])

        # display 4th image
        self.win.flip()
        thisTrial["Stimulus 4 Onset"] = self.masterTimer.clock.getTime()

        # prep ISI
        self.draw_fix()

        self.masterTimer.wait(exp_info["stimTime"])

        # ISI
        self.win.flip()

        # prep 5th image
        self.draw_fix()
        self.draw_stimuli_pres(thisTrial, 4)

        self.masterTimer.wait(thisTrial["ISIs"][3])

        # display 5th image
        self.win.flip()
        thisTrial["Stimulus 5 Onset"] = self.masterTimer.clock.getTime()

        # prep ISI
        self.draw_fix()

        self.masterTimer.wait(exp_info["stimTime"])

        # ISI
        self.win.flip()

        # prep 6th image
        self.draw_fix()
        self.draw_stimuli_pres(thisTrial, 5)

        self.masterTimer.wait(thisTrial["ISIs"][4])

        # 6th image
        self.win.flip()
        thisTrial["Stimulus 6 Onset"] = self.masterTimer.clock.getTime()

        # prep 6th ITI + Delay
        self.draw_fix()

        self.masterTimer.wait(exp_info["stimTime"])


        # last ISI + delay
        self.win.flip()

        # prep test probe
        self.draw_fix()
        thisTrial["test_image"].draw()

        self.masterTimer.wait(thisTrial["ISIs"][5] + exp_info["delayTime"])

        # test probe
        self.win.flip()
        thisTrial["Test Onset"] = self.masterTimer.clock.getTime()
        self.kb.clearEvents()
        self.kb.clock.reset()
        keys = self.kb.getKeys(keyList=exp_info["keys"], clear=True)
        while (not keys) and (self.kb.clock.getTime() < exp_info["ResponseWindow"]):
            keys = self.kb.getKeys(keyList=exp_info["keys"], clear=True)
        # update the ideal time to be either now or the end of the response window
        if self.kb.clock.getTime() < exp_info["ResponseWindow"]:
            self.masterTimer.wait(exp_info["ResponseWindow"])  # wait out the difference
        else:
            self.masterTimer.update_ideal(exp_info["ResponseWindow"])  # else just assume it's been the appropriate amount of time

        # ITI
        self.draw_fix()
        self.win.flip()

        # record the response
        if keys:
            thisTrial["Response"] = keys[0].name
            thisTrial["RT"] = keys[0].rt
            thisTrial["Accuracy"] = thisTrial["Response"] == thisTrial["correct_response"]
        else:
            thisTrial["Response"] = None
            thisTrial["RT"] = None
            thisTrial["Accuracy"] = None

        self.data.append(thisTrial)
        self.masterTimer.wait(thisTrial["ITI"])

    def run(self):

        self.get_run_info_and_file()

        self.init_rng()

        self.init_stimuli()

        self.trials = self.make_block()

        # PREPARE TO START THE RUN - THE SCANNER WILL TRIGGER
        self.win.mouseVisible=False
        text_stim = visual.TextStim(
            self.win,
            text="Ready to scan.",
            pos=(0.5, 0),
            units="norm",
            alignHoriz="center",
            alignVert="center"
        )
        text_stim.draw()
        self.win.flip()
        _ = event.waitKeys(keyList=["t", "T"])

        # Initialize Master Timestamp once the scanner starts, wait some seconds
        self.masterTimer = masterClock()
        self.draw_fix()
        self.win.flip()
        self.masterTimer.wait(self.exp_info["initialWait"])  # TRs to skips

        # run the task, get the data!
        self.data = []
        for thisTrial in self.trials:
            self.run_trial(thisTrial)

        self.masterTimer.wait(self.exp_info["finalCoolDown"])

        # save the data
        df = pd.DataFrame(self.data)
        df.to_csv("data/{}.tsv".format(self.filename))
        self.win.mouseVisible=True

        core.quit()


if __name__ == "__main__":
    task = Task(scan_monitor_details if SCANNING else beh_monitor_details,
                exp_info)
    task.run()
