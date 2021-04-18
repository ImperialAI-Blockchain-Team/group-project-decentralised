import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0x9b673923B47B678299834c82f63049E46f6b3D8b';

export default new web3.eth.Contract(registry, address);