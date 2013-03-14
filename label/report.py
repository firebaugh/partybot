'''
Caitlyn Clabaugh

Simple reporter for batch mashup labeling.
Takes in output .dat file from GA labeling.
Emails concise update of batch's progress.
'''

from optparse import OptionParser
import smtplib, string

def main():
    parser = OptionParser(usage="usage: %prog results.dat")
    (options, args) = parser.parse_args()
    # get .dat filename
    if len(args) < 1:
        parser.print_help()
        exit(0)
    filename = args[0]

    # read .dat
    try:
        f = open(filename, "r")
        lines = f.readlines()
        subject = string.split(lines[0]," ").pop()
        run = "".join(lines[0:6])
        best = lines.pop()
        update = "\n".join([run,best])
        f.close()
    except:
        subject = "failure"
        update = "No "+filename+" generated. Something has gone wrong."

    # create email
    FROM = 'hell.week.help@gmail.com'
    TO = ['ceclabaugh@gmail.com']
    USR = 'hell.week.help'
    PWD = 'HelpMePlease!'
    SUBJECT = "GA Progress UPDATE: "+subject
    TEXT = update
    BODY = string.join((
        "From: %s" % FROM,
        "To: %s" % TO,
        "Subject: %s" % SUBJECT ,
        "",
        TEXT
        ), "\r\n")
            
    # send update
    server = smtplib.SMTP( "smtp.gmail.com", 587 )
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login( USR, PWD )
    server.sendmail(FROM, TO, BODY)
    server.close()


if __name__ == "__main__":
    main()
