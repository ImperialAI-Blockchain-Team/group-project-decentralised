import web3 from './web3';
import {registry} from "./abi/abi";

const address = '0x701f2bd3Be4995415E72553a565Bda253dfA68d2';


export default new web3.eth.Contract(registry, address);