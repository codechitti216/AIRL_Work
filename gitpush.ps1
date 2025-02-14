param (
    [string]$action,
    [string]$message
)

function RepoAction {
    param (
        [string]$repo,
        [string]$message
    )

    # Check for changes
    $status = git -C $repo status --porcelain
    if ($status) {
        Write-Host "✅ Pushing changes in $repo..."
        git -C $repo add .
        git -C $repo commit -m "$message"
        git -C $repo push origin $(git -C $repo symbolic-ref --short HEAD)
    } else {
        Write-Host "⚠️ No changes in $repo, skipping..."
    }
}

switch ($action) {
    "apush" { RepoAction "." "$message" }
    "spush" { RepoAction "SonarNet" "$message" }
    "bpush" { RepoAction "." "$message"; RepoAction "SonarNet" "$message" }
    default { Write-Host "❌ Usage: git {apush|spush|bpush} `"Commit message`"" }
}
