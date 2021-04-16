import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0x3f8C6af38C3D9807e164392d603b12bEdf65976c';

export default new web3.eth.Contract(registry, address);