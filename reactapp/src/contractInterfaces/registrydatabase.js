import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0x52A5416e00CF724387DD59103F4bB7495c88d98c';

export default new web3.eth.Contract(registry, address);
