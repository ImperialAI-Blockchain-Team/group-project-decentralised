import web3 from './web3';
import {registry} from "../abi/abi";

const address = '0xF7c74299814Ee2B17552a793AAcda10F3298ED8A';

export default new web3.eth.Contract(registry, address);