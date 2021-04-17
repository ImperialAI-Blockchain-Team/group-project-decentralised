import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0xEBC095934C0899652E1d528213A0940D87E434dc';

export default new web3.eth.Contract(registry, address);