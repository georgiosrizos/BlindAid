//
// Authors:  Dan Iorga, Georgios Rizos, Georgios Theodorakis, Johannes Wiebe, Thomas Uriot
//
// BlindAid: HiPEDS CDT group project - cohort 2017 - Imperial College London
//

// User micro:bit / smart sensor code.
// Receives measurements from the 4 transmitters, performs running average filtering and sends data to the
// micro:bit connected to the computer.

let degrees: number
let id = 5
let Nm = 150
let Nmb = 5
let max_queue_size= 10
let freq = 0

radio.setTransmitPower(7)
radio.setGroup(22)
input.calibrateCompass()
basic.showNumber(id)

let meas_list: Array<number>
let meas_copy: Array<number>
let meas_param = 20

input.onButtonPressed(Button.A, () => {
    meas_param = (meas_param + 10) % 100
    basic.showNumber(meas_param)
})

meas_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

// Performs running average filtering using an exponential filter.
function ema(old_avg: number, new_data: number, param: number) {
    return (new_data - old_avg) * param / 100 + old_avg;
}

// Reads incoming messages and sends them to the computer
radio.onDataPacketReceived(({ signal, receivedString, receivedNumber }) => {
    if (receivedString == "1") {
        meas_list[1] = ema(meas_list[1], signal * 10, meas_param)
        meas_list[5] = ema(meas_list[5], receivedNumber * 10, meas_param)
    }
    else if (receivedString == "2") {
        meas_list[2] = ema(meas_list[2], signal * 10, meas_param)
        meas_list[6] = ema(meas_list[6], receivedNumber * 10, meas_param)
    }
    else if (receivedString == "3") {
        meas_list[3] = ema(meas_list[3], signal * 10, meas_param)
        meas_list[7] = ema(meas_list[7], receivedNumber * 10, meas_param)
    }
    else if (receivedString == "4") {
        meas_list[4] = ema(meas_list[4], signal * 10, meas_param)
        meas_list[8] = ema(meas_list[8], receivedNumber * 10, meas_param)
    }
    // this case is for alerting the user about the directions
    else if (receivedString == "6") {
        // Our convention about directions is
        // assume 0 == left, 1 == straight, 2 == right
        if (receivedNumber == 0) {
            freq = 262
            music.playTone(freq, 100)
            music.playTone(freq, 100)
        }
        else if (receivedNumber == 1) {
            // no sound if the direction is correct
        }
        else if (receivedNumber == 2) {
            freq = 262
            music.playTone(freq, 1000)
        }
    }
    else {
        // TODO: handle this case
    }
    meas_list[9] = ema(meas_list[9], input.acceleration(Dimension.X), meas_param)
    meas_list[10] = ema(meas_list[10], input.acceleration(Dimension.Y), meas_param)
    meas_list[11] = ema(meas_list[11], input.acceleration(Dimension.Z), meas_param)

    meas_list[12] = ema(meas_list[12], input.magneticForce(Dimension.X) * 10, meas_param)
    meas_list[13] = ema(meas_list[13], input.magneticForce(Dimension.Y) * 10, meas_param)
    meas_list[14] = ema(meas_list[14], input.magneticForce(Dimension.Z) * 10, meas_param)

    // TODO: BUG - Averaging 360 and 0 degrees returns values close to 180.
    meas_list[15] = ema(meas_list[15], input.compassHeading() * 10, meas_param)
})


let msg_id = 0
let val_id: string

// Every 10 ms, send the current value estimates to the micro:bit connected to the computer.
basic.forever(() => {
    msg_id = input.runningTime()
    meas_copy = meas_list
    for (let i = 1; i < 16; i++) {
        val_id = msg_id.toString() + "_" + i.toString()
        radio.sendValue(val_id, meas_copy[i])
        basic.pause(10)
    }
})
