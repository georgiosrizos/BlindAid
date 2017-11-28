//
// Authors:  Dan Iorga, Georgios Rizos, Georgios Theodorakis, Johannes Wiebe, Thomas Uriot
//
// BlindAid: HiPEDS CDT group project - cohort 2017 - Imperial College London
//

// Code for one of the four transmitting micro:bits to be placed in the corner.
// Periodically gets a sonar signal and transmits it to the receiver.

let id = 1
let id_str = "1"
let sp = 0

radio.setTransmitPower(7)
radio.setGroup(22)
basic.showNumber(id)

// Sets the ID of the microbit - cycles through 1-4.
input.onButtonPressed(Button.B, () => {
    if (id < 4) {
        id += 1
    } else {
        id = 1
    }
    basic.showNumber(id)
})

// Every 10 ms, get a sonar measurement and transmit it along with ID via radio.
while (true) {
    sp = sonar.ping(DigitalPin.P0, DigitalPin.P1, PingUnit.Centimeters)
    radio.sendNumber(id)
    id_str = id.toString()
    radio.sendValue(id_str, sp)

    basic.pause(10)
}
