if [ "$1" = "S" ]; then
    git --git-dir=SonarNet/.git --work-tree=SonarNet add .
    git --git-dir=SonarNet/.git --work-tree=SonarNet commit -m "$2"
    git --git-dir=SonarNet/.git --work-tree=SonarNet push origin main
elif [ "$1" = "A" ]; then
    git add .
    git commit -m "$2"
    git push origin $(git branch --show-current)
else
    echo "Usage: ./push.sh [S|A] \"Commit Message\""
fi
