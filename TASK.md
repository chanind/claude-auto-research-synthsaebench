# Task: Improve SAEs using SynthSAEBench

Your goal is to try to build an SAE architecture that performs as well as possible on the SynthSAEBench-16k model.

You must abide by the following:

- 200M samples must be used to train the SAE
- Learning rate of 3e-4
- The SAE must have width 4096
- the model must be 'decoderesearch/synth-sae-bench-16k-v1'

While iterating, you can of course modify any of the above, but for a run to count it must follow those constraints.

We mainly care about maximizing the following 2 SAE quality metrics:

- F1 score
- MCC

We do not care about variance explained, so don't worry if that metric is not optimal.

We expect that an SAE L0 around 25 is the most likely to be successful, since that's the L0 of the first 4096 features of the synthetic model, but this is just an idea and if you find better results with a differnent L0, that's fine.

## Documentation

SynthSAEBench is built-in to SAELens, and docs are available at https://decoderesearch.github.io/SAELens/latest/synth_sae_bench/ and https://decoderesearch.github.io/SAELens/latest/synthetic_data/

There's also a paper about SynthSAEBench on arxiv at: https://arxiv.org/abs/2602.14687v1

## Workflow

You should work on a research sprint, where you first brainstorm ideas and search for related work that might be relevant, then pick an idea to investigate and implement it, then iterate on it and see how much progress you can make. Keep a research log as you go in the dir `synthsae-research/`. For your sprint idea, you should create a new dir in that folder prefixed by the date and start time of the sprint in UTC (e.g. 2026-02-19--10-37) and suffixed with the name of the idea as a title. Put all your code in that dir as well, give it a README.md that describes the idea and the progress you made, and put any plots or data you generate in that dir as well. You should only work on 1 idea, and when you are done you can exit. This will run in a loop so each iteration is a single idea. When you're done, please also make a PDF report of your research and put it in the same dir as the code including plots and tables as well as the contents of the README.md.

When you start, the first thing you should do is make the README.md in the sprint dir and describe the idea you're going to try, the motivation behind it, and what you expect to happen. This will help make sure that other agents don't duplicate your work while you're working on it.

Check the experiment results in the `synthsae-research/` dir, have a look at them first so you don't duplicate work that's already been tried (or is in-progress), and to get some ideas of what to try next. Also keep notes as you do in the your research log, for instance explain as well the rationale behind why you picked the idea you did, and any insights you learn as you go in your README.md.

When you are done, add a DONE.txt file to the dir to indicate that the sprint is complete.

NOTE: If you see existing experiments that do not yet have a DONE.txt file, do not work on them, this means a different agent is already working on it.

## Debugging

One thing you should try as you go is to understand why your SAEs work or don't work by comparing the trained SAE dictionary to the SyntheticModel dictionary, and get a sense of what is messing up the SAE. For instance is it that SAE latents tracking features that have high firing variance are getting messed up? Or is the SAE mixing features together based on their superposition noise or firing correlation or something? For firing frequencies, keep in mind that the raw firing probabilities on the model are not the true firing probabilities due to the hierarchy correction (you may need to experimentally determine the true firing probabilities).

## Some ideas to try

Below are just some ideas, you don't need to do these but just thought I'd share some for inspiration:

- Try setting `matryoshka_loss_multipliers` on XMatryoshkaBatchTopKTrainingSAE to something other than 1.0. Based on feature hedging paper
- Try settings `detach_matryoshka_losses=True`, also try with `matryoshka_loss_multipliers` set to something other than 1.0.
- Try creating a block Matching Pursuit SAE where instead of picking 1 latent at a time, it picks a block of N latents at a time.
- Try the above, but make the block size randomly vary between 1 and N or something similar.
- Try some traditional dictionary learning algorithms, like K-SVD or Orthogonal Matching Pursuit or something else (might be hard to get these to run efficiently enough to be feasible though)
- Research other fields like compressed sensing / sparse coding / ICA / LASSO / etc... see if there are any ideas there that we could try out
- Try variations on an MLP encoder. maybe start with a resnet like we have currently in the code, or think through what might help the SAE learn better features without going off the rails.
- Try googling for interesting research papers and ideas on the internet that could be relevant to the task.
- Try mixing adding an auxiliary TopK loss to a BatchTopK or Matryoshka BatchTopK SAE to see if that helps it learn better features. (there are some papers saying TopK SAEs are great for identifying unique features)

## Have fun!

We expect that not every idea will be successful, and that's fine. There's a lot to learn from things that don't work too. So if you don't make progress on an idea, that's fine, but you should still document what you tried and what you learned.
