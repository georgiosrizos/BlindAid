//
// Authors:  Dan Iorga, Georgios Rizos, Georgios Theodorakis, Johannes Wiebe, Thomas Uriot
//
// BlindAid: HiPEDS CDT group project - cohort 2017 - Imperial College London
//

// Code for the connected to the computer micro:bit.
// Receives data from user micro:bit and then sends it to the computer via serial.
// Also receives direction commands from computer and sends it to the user.

let id_str = '6'
let command: number
let cp = 0

radio.setTransmitPower(7)
radio.setGroup(22)
basic.showString(id_str)

basic.showNumber(cp)

// Write data using the Serial API (feature_id, feature_value).
radio.onDataPacketReceived(({ receivedNumber, receivedString }) => {

    if (receivedString !== "") {
        serial.writeString(receivedString + ":" + receivedNumber.toString() + "\n")
    }
})

// Send the next audio command to user's micro:bit with Radio API.
serial.onDataReceived(serial.delimiters(Delimiters.NewLine), () => {
    command = parseInt( serial.readUntil(serial.delimiters(Delimiters.NewLine)))
    radio.sendValue(id_str, command)
})

// On button press, set checkpoint.
input.onButtonPressed(Button.A, () => {
    cp += 1
    serial.writeString("-1_0:" + cp.toString() + "\n")
    basic.showNumber(cp)
})
