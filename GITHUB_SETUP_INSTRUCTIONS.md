# GitHub Repository Setup Instructions

Follow these steps to launch https://github.com/cputer/mind

1. Create repo (already created): **github.com/cputer/mind**
2. Clone & push:
```bash
git clone https://github.com/cputer/mind.git
cd mind
# add files from this package if needed, then:
git add .
git commit -m "Initial commit: MIND language v0.1.0"
git push origin main
```
3. Configure:
   - Add topics: ai-language, machine-learning, compiler, tensor-native, autodiff, mlir, gpu-computing, programming-language, deep-learning, parallel-computing, systems-programming
   - Enable Issues, Discussions, Projects
   - Social preview image (1280x640)
4. Labels (with GitHub CLI):
```bash
gh label create "good first issue" --color "7057ff" --description "Good for newcomers"
gh label create "💰 bounty" --color "0e8a16" --description "Eligible for bounty"
gh label create "translation" --color "c5def5" --description "Help translate documentation"
gh label create "help wanted" --color "008672" --description "Extra attention needed"
gh label create "enhancement" --color "a2eeef" --description "New feature or request"
gh label create "bug" --color "d73a4a" --description "Something isn't working"
gh label create "documentation" --color "0075ca" --description "Improvements to docs"
gh label create "RFC" --color "d4c5f9" --description "Request for Comments"
```
