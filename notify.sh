# http://apple.stackexchange.com/questions/57412/how-can-i-trigger-a-notification-center-notification-from-an-applescript-or-shel

# Run after commands to trigger notification after job has completed. e.g.
# $ job-that-takes-a-long-time ; source notify.sh
osascript -e 'display notification "Job complete!" with title "Alert"'
