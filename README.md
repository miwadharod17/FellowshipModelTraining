# Virtual Lab Model Training

Repo: https://github.com/miwadharod17/FellowshipModelTraining
Python: **3.12**

## 1. Clone and set up

```bash
git clone https://github.com/miwadharod17/FellowshipModelTraining.git
cd FellowshipModelTraining
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m tests.test_packs      # should print PASS
python -m tests.test_data       # should print PASS
```

## 2. Create your own branch

Use your own name or track, for example `person_a`, `person_b`, `person_c`.

```bash
git checkout main
git pull
git checkout -b person_x
git push -u origin person_x
```

## 3. Work on your branch

Run scripts from the project root, for example `python -m train.train_dl5`.

```bash
git status
git add <your files>
git commit -m "Short message saying what you did"
git push
```

Before working, fetch the changes

```bash
git fetch origin
git merge origin/main
```

## 4. Merge into main when finished

1. Make sure your branch is up to date and the tests pass:
   ```bash
   git fetch origin
   git merge origin/main
   python -m tests.test_packs && python -m tests.test_data
   git push
   ```
2. On GitHub, open **Pull requests -> New pull request**.
3. Set base to `main` and compare to your branch, then click **Create pull request**.
4. Once a teammate has looked it over, click **Merge pull request**.
5. Update your local copy:
   ```bash
   git checkout main
   git pull
   ```

Without GitHub pull requests, merge locally instead:

```bash
git checkout main
git pull
git merge person_x
git push
```

## Rules

- Seed is 42 (`from common.seed import set_seed`).
- Never re-split data. Load the saved files in `data/splits/`.
- Names use lowercase letters, digits and underscores only.
- Never edit another person's files or the shared `common/` folder without telling the group.
